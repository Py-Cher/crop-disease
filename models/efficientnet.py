import torch
from torch import nn
from torchvision.models import efficientnet_b7, efficientnet_b1
import torch.nn.functional as F

class efficientnet_b7_base(nn.Module):
    def __init__(
        self,
        crop_output_dim = 6,
        disease_output_dim = 12,
        risk_output_dim = 4, 
        ):
        
        super().__init__()
        
        self.model = efficientnet_b7(pretrained=True)
        self.crop = nn.Linear(1000,crop_output_dim)
        self.disease = nn.Linear(1000 + crop_output_dim, disease_output_dim)
        self.risk = nn.Linear(1000 + disease_output_dim, risk_output_dim)
        
    def forward(self,image):
        x = F.relu(self.model(image))
        
        crop_pre = self.crop(x)
        disease_pre = self.disease(torch.cat([x,F.relu(crop_pre.clone())],dim=1))
        risk_pre = self.risk(torch.cat([x,F.relu(disease_pre.clone())], dim=1))
        
        return crop_pre, disease_pre, risk_pre
    
    
    
class efficientnet_b1_base(nn.Module):
    def __init__(
        self,
        crop_output_dim = 6,
        disease_output_dim = 12,
        risk_output_dim = 4, 
        ):
        
        super().__init__()
        
        self.model = efficientnet_b1(pretrained=True)
        self.crop = nn.Linear(1000,crop_output_dim)
        self.disease = nn.Linear(1000 + crop_output_dim, disease_output_dim)
        self.risk = nn.Linear(1000 + disease_output_dim, risk_output_dim)
        
    def forward(self,image):
        x = F.relu(self.model(image))
        
        crop_pre = self.crop(x)
        disease_pre = self.disease(torch.cat([x,F.relu(crop_pre.clone())],dim=1))
        risk_pre = self.risk(torch.cat([x,F.relu(disease_pre.clone())], dim=1))
        
        return crop_pre, disease_pre, risk_pre