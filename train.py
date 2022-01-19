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
from sklearn.model_selection import KFold, StratifiedKFold

from models.efficientnet import efficientnet_b7_base, efficientnet_b1_base
from dataloader import ImageDataset
from utils import *

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
    parser.add_argument("--valid_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--strategy", type=str, default="step")
    parser.add_argument("--ckt_folder", type=str, default="./")
    parser.add_argument("--img_folder", type=str, default="./")
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--reload_step_from", type=int, default=0)
    parser.add_argument("--reload_folder_from", type=int, default=0)
    parser.add_argument("--model_name",type=str,default='')

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    data = pd.read_csv(args.train_csv)
    label = data['label'].apply(lambda x: x.split('_')[0]) # crop label이 아닌 전체 라벨 대상으로도 해보기
    kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True)
    
    
    if args.reload_step_from:
        print(f'starting from {args.reload_folder_from} folder, {args.reload_step_from} step')
    
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(data,label)):     
            
        if args.reload_folder_from > fold:
            continue
            
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_dataset = ImageDataset(args.img_folder, data.iloc[train_ids], mode = 'train')        
        valid_dataset = ImageDataset(args.img_folder, data.iloc[valid_ids], mode = 'test')
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        )
        
        model = efficientnet_b1_base()
        model.apply(reset_weights)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        criterion3 = nn.CrossEntropyLoss()
            
        n_iters = len(train_dataloader)

        if args.strategy == "epoch":
            args.valid_every = n_iters
            args.save_every = n_iters
            
        itr_global = 0
        start_itr = 0
        if fold == args.reload_folder_from and args.reload_step_from:
            itr_global = args.reload_step_from
            start_itr = args.reload_step_from

        for epoch in range(int(start_itr / n_iters) + 1, args.total_epoch + 1):
            itr_start_time = time.time()
            losses1 = []
            losses2 = []
            losses3 = []
            
            for batch in train_dataloader:
                model.train()
                optimizer.zero_grad()
                
                input = batch['input'].to(device)
                crop_label = torch.LongTensor(batch['crop']).to(device)
                disease_label = torch.LongTensor(batch['disease']).to(device)
                risk_label = torch.LongTensor(batch['risk']).to(device)
                
                crop_pre, disease_pre, risk_pre = model(input)
                
                loss1 = criterion1(crop_pre, crop_label)
                loss2 = criterion2(disease_pre, disease_label) 
                loss3 = criterion3(risk_pre, risk_label)
                
                loss = loss1 + loss2 + loss3
                loss.backward()
                
                losses1.append(loss1.item())
                losses2.append(loss2.item())
                losses3.append(loss3.item())
                
                optimizer.step()
                
                if itr_global % args.log_every == 0:
                        elapsed = time.time() - itr_start_time
                        print(
                            "epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss1=%.5f Loss2=%.5f Loss3=%.5f Loss=%.5f"
                            % ( 
                                epoch,
                                args.total_epoch,
                                itr_global % n_iters,
                                n_iters,
                                elapsed,
                                np.mean(losses1),
                                np.mean(losses2),
                                np.mean(losses3),
                                np.mean(losses1+losses2+losses3),
                            )
                        )
                        
                        losses1 = []
                        losses2 = []
                        losses3 = []
                        itr_start_time = time.time()
                    
                itr_global = itr_global + 1
                
                if itr_global % args.valid_every == 0:
                    print("validating..")
                    model.eval()
                    
                    losses1 = []
                    losses2 = []
                    losses3 = []
                    correct, total = [0, 0, 0], [0, 0, 0]
                    
                    with torch.no_grad():
                        for j, batch in enumerate(valid_dataloader):
                            print("validation:" + str(j) + "/" + str(len(valid_dataloader)))
                            input = batch['input'].to(device)
                            crop_label = batch['crop'].to(device)
                            disease_label = batch['disease'].to(device)
                            risk_label = batch['risk'].to(device)
                            
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
                            
                        print(
                            "epo:[%d/%d] itr:[%d/%d] Loss1=%.5f Loss2=%.5f Loss3=%.5f Loss=%.5f Acc1=%.3f Acc2=%.3f Acc3=%.3f Acc=%.3f"
                            % ( 
                                epoch,
                                args.total_epoch,
                                itr_global % n_iters,
                                n_iters,
                                np.mean(losses1),
                                np.mean(losses2),
                                np.mean(losses3),
                                np.mean(losses1+losses2+losses3),
                                100.0 * (correct[0] / total[0]),
                                100.0 * (correct[1] / total[1]),
                                100.0 * (correct[2] / total[2]),
                                100.0 * (sum(np.sum(correct)) / sum(np.sum(total))),
                            )
                        )
                  
                if itr_global % args.save_every == 0:          
                    dict_for_infer = {
                            "model": model.state_dict(),
                            "opt": optimizer.state_dict(),
                            #"scaler": scheduler.state_dict(),
                            #"amp": amp.state_dict(),
                            "batch_size": args.batch_size,
                            "epochs": args.total_epoch,
                            "learning_rate": args.lr,
                            "device": device,
                        }
                    
                    print("saving...")
                    
                    os.makedirs(args.ckt_folder, exist_ok=True)
                    save_dir = os.path.join(args.ckt_folder, f"{args.model_name}-checkpoint_{fold}fold_{epoch}epoch_{itr_global}step")
                    
                    torch.save(dict_for_infer, save_dir)

                    with open(os.path.join(args.ckt_folder, "dict_for_infer"), "wb") as f:
                        pickle.dump(dict_for_infer, f)

                    print("저장 완료!")

