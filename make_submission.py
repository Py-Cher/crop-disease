import pandas as pd
from tqdm import tqdm 
import time
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
from models.efficientnet import efficientnet_base
from dataloader import ImageDataset_test
from utils import custom_beam_search_decoder

def _rebuild_xla_tensor(data, dtype, device, requires_grad):
  tensor = torch.from_numpy(data).to(dtype=dtype, device='cpu')
  tensor.requires_grad = requires_grad
  return tensor

#torch._utils._rebuild_xla_tensor = _rebuild_xla_tensor

#device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
device = xm.xla_device()
model = efficientnet_base(base_model='efficientnet_b5',base_model_ouput_dim=1000)
model.to(device)

checkpoint = torch.load('../checkpoint/test-checkpoint_3fold_200epoch')
model.load_state_dict(checkpoint['model'])
submission = pd.read_csv('../LG_data/sample_submission.csv')

test_dataset = ImageDataset_test('../LG_data/test',submission,mode = 'test')
test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

disease_encoder = {'00':0, 'a11':1, 'a12':2, 'a5':3, 'a7':4, 'a9':5, 'b3':6, 'b4':7, 'b5':8, 'b6':9, 'b7':10, 'b8':11}
disease_decoder= dict(map(reversed,disease_encoder.items()))
decoder = custom_beam_search_decoder()

crop_result = []
disease_result = []
risk_result = []

model.eval()
with torch.no_grad():
    for step, batch in tqdm(enumerate(test_dataloader, start=1)):
        input = batch['input'].to(device)
        crop_pre, disease_pre, risk_pre = model(input)

        crop_result.extend(crop_pre.detach().cpu().numpy())
        disease_result.extend(disease_pre.detach().cpu().numpy())
        risk_result.extend(risk_pre.detach().cpu().numpy())

result = []
for value in zip(crop_result, disease_result, risk_result):
  result.append(decoder.decode(value,[2,3,2]))

final_result = []
for row in result:
  final_result.append(row[0][0])

final_prediction = []
for a,b,c in final_result:
    final_prediction.append(f'{a+1}_{disease_decoder[b]}_{c}')
        
result_file = pd.read_csv('../LG_data/sample_submission.csv')
result_file['label'] = pd.Series(final_prediction)
result_file.to_csv('../result.csv',index=False)