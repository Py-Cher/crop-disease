import torch
from torch import nn
import torch.nn.functional as F
#import timm
from torchvision.models import efficientnet_b5

class efficientnet_base(nn.Module):
    def __init__(
        self,
        base_model = 'efficientnet_b5',
        base_model_ouput_dim = 1000,
        crop_output_dim = 6,
        disease_output_dim = 12,
        risk_output_dim = 4, 
        ):
        
        super().__init__()
        
        #self.model = timm.create_model(base_model, pretrained=True, num_classes=base_model_ouput_dim)
        self.model = efficientnet_b5(pretrained=True)
        self.crop = nn.Linear(base_model_ouput_dim,crop_output_dim)
        self.disease = nn.Linear(base_model_ouput_dim + crop_output_dim, disease_output_dim)
        self.risk = nn.Linear(base_model_ouput_dim + disease_output_dim, risk_output_dim)
        
    def forward(self,image):
        x = F.relu(self.model(image))
        
        crop_pre = self.crop(x)
        disease_pre = self.disease(torch.cat([x,F.relu(crop_pre.clone())],dim=1))
        risk_pre = self.risk(torch.cat([x,F.relu(disease_pre.clone())], dim=1))
        
        return F.log_softmax(crop_pre, dim=-1), F.log_softmax(disease_pre, dim=-1), F.log_softmax(risk_pre, dim=-1)
    