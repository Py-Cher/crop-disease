import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(
        self,
        image_root_path,
        dataframe,
        mode = 'train',
    ):
        
        super().__init__()
        self.path = image_root_path
        self.data = dataframe
        self.data['image'] = self.data['image'].apply(lambda x: os.path.join(self.path,str(x),str(x)))
        self.mode = mode
        
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize([384, 512]),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        else:
            self.transforms = transforms.Compose([
                transforms.Resize([384, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.disease_encoder = {'00':0, 'a11':1, 'a12':2, 'a5':3, 'a7':4, 'a9':5, 'b3':6, 'b4':7, 'b5':8, 'b6':9, 'b7':10, 'b8':11}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]['image'] + '.jpg')
        crop, disease, risk = self.data.iloc[idx]['label'].split('_')
        disease = self.disease_encoder[disease]
        
        image = self.transforms(image)
        
        return {'input':image, 'crop':int(crop) - 1, 'disease':int(disease), 'risk':int(risk)}
        