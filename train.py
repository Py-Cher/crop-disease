import argparse
import pandas as pd
import time
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold

from models.efficientnet import efficientnet_base
from dataloader import ImageDataset
from utils import *

def train_loop(args, writer, model, optimizer, criterion1, criterion2, criterion3, train_dataloader, epoch, device):
    itr_start_time = time.time()
    n_iters = len(train_dataloader)
    
    losses1 = []
    losses2 = []
    losses3 = []
    
    for step, batch in enumerate(train_dataloader, start=1):
        model.train()
        optimizer.zero_grad()
        
        input = batch['input']
        crop_label = batch['crop'].squeeze()
        disease_label = batch['disease'].squeeze()
        risk_label = batch['risk'].squeeze()
        
        crop_pre, disease_pre, risk_pre = model(input)
        
        loss1 = criterion1(crop_pre, crop_label)
        loss2 = criterion2(disease_pre, disease_label) 
        loss3 = criterion3(risk_pre, risk_label)
        
        loss = loss1 + loss2 + loss3
        loss.backward()
        
                
        xm.optimizer_step(optimizer)
        #optimizer.step()
        
        losses1.append(loss1.item())
        losses2.append(loss2.item())
        losses3.append(loss3.item())
        
        if step % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            xm.add_step_closure(
                train_logging,      
                args=(writer,epoch,args.total_epoch,step,n_iters,elapsed, losses1, losses2,losses3),
                run_async=True
            )
            
            losses1 = []
            losses2 = []
            losses3 = []
            itr_start_time = time.time()

def validation_loop(args, writer, model, criterion1, criterion2, criterion3, valid_dataloader, epoch, device):
    model.eval()
    
    losses1 = []
    losses2 = []
    losses3 = []
    correct, total = [0, 0, 0], [0, 0, 0]
    n_iters = len(valid_dataloader)
    
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader, start=1):
            input = batch['input']
            crop_label = batch['crop'].squeeze()
            disease_label = batch['disease'].squeeze()
            risk_label = batch['risk'].squeeze()
            
            crop_pre, disease_pre, risk_pre = model(input)
            
            loss1 = criterion1(crop_pre, crop_label)
            loss2 = criterion2(disease_pre, disease_label) 
            loss3 = criterion3(risk_pre, risk_label)
            
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            
            _, predicted = torch.max(crop_pre.data, 1)
            total[0] += crop_label.size(0)
            correct[0] += (predicted == crop_label).sum().item()
            
            _, predicted = torch.max(disease_pre.data, 1)
            total[1] += disease_label.size(0)
            correct[1] += (predicted == disease_label).sum().item()
            
            _, predicted = torch.max(risk_pre.data, 1)
            total[2] += risk_label.size(0)
            correct[2] += (predicted == risk_label).sum().item()
        
            xm.add_step_closure(
                valid_logging,
                args=(writer, epoch,args.total_epoch,step,n_iters, losses1, losses2,losses3,correct,total),
                run_async=True
            )
                
            losses1 = []
            losses2 = []
            losses3 = []    
            correct, total = [0, 0, 0], [0, 0, 0]
            
def save_model(args, model, optimizer, fold, epoch):
    dict_for_infer = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        #"scaler": scheduler.state_dict(),
        #"amp": amp.state_dict(),
        "batch_size": args.batch_size,
        "epochs": args.total_epoch,
        "learning_rate": args.lr,
    }
    
    os.makedirs(args.ckt_folder, exist_ok=True)
    save_dir = os.path.join(args.ckt_folder, f"{args.model_name}-checkpoint_{fold}fold_{epoch}epoch")
    
    xm.save(dict_for_infer, save_dir)

    '''with open(os.path.join(args.ckt_folder, "dict_for_infer"), "wb") as f:
        pickle.dump(dict_for_infer, f)

    print("save complete")'''

    
def main(index, args):    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    device = xm.xla_device()
    
    data = pd.read_csv(args.train_csv)
    label = data['label'].values #apply(lambda x: x.split('_')[0]) # try crop_dis_risk label instead of crop label
    kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True)
    
    if args.reload_epoch_from:
        print(f'starting from {args.reload_folder_from} folder, {args.reload_epoch_from} epoch {args.reload_model_from} model')
    
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(data,label)):     
        
        if args.reload_folder_from > fold:
            continue
            
        writer = None
        if xm.is_master_ordinal():
            writer = test_utils.get_summary_writer(f'runs/{args.model_name}-{fold}fold')
            
            print(f'FOLD {fold}')
            print('--------------------------------')

        train_dataset = ImageDataset(args.img_folder, data.iloc[train_ids], mode = 'train')        
        valid_dataset = ImageDataset(args.img_folder, data.iloc[valid_ids], mode = 'valid')
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
    
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        
        train_dataloader = pl.MpDeviceLoader(
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=32,
                sampler=train_sampler,
                pin_memory=True,
                drop_last=True,
            ), device
        )
        
        valid_dataloader = pl.MpDeviceLoader(
            DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                num_workers=32,
                sampler=valid_sampler,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            ), device
        )
        
        model = efficientnet_base(base_model=args.model,base_model_ouput_dim=1000)
        model.apply(reset_weights)
        model.to(device)
        
        lr = args.lr * xm.xrt_world_size()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion1 = nn.NLLLoss()
        criterion2 = nn.NLLLoss()
        criterion3 = nn.NLLLoss()
            
        start_epoch = 0
        if fold == args.reload_folder_from and args.reload_epoch_from:
            if args.reload_model_from == "":
                raise('need model file name')
            
            print('loading model')
            start_epoch = args.reload_epoch_from
            save_dir = os.path.join(args.ckt_folder, args.reload_model_from)
            checkpoint = torch.load(save_dir)
            optimizer.load_state_dict(checkpoint['opt'])
            model.load_state_dict(checkpoint['model'])
            

        for epoch in range(start_epoch, args.total_epoch + 1):
            train_loop(args, writer, model, optimizer, criterion1, criterion2, criterion3, train_dataloader, epoch, device)
            validation_loop(args, writer, model, criterion1, criterion2, criterion3, valid_dataloader, epoch, device)          
            if epoch % args.save_every == 0:
                if xm.is_master_ordinal():
                    xm.add_step_closure(
                        save_model,
                        args = (args, model, optimizer, fold, epoch),   
                        run_async= False
                    )
                    
                    print('saved')
                       
        test_utils.close_summary_writer(writer)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_epoch", type=int, default=40)
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    #parser.add_argument("--valid_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--ckt_folder", type=str, default="./")
    parser.add_argument("--img_folder", type=str, default="./")
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--reload_epoch_from", type=int, default=0)
    parser.add_argument("--reload_folder_from", type=int, default=0)
    parser.add_argument("--reload_model_from", type=str, default='') 
    parser.add_argument("--model",type=str,default='',required=True)
    parser.add_argument("--model_name",type=str,default='',required=True)

    args = parser.parse_args()

    xmp.spawn(main, args=(args,))