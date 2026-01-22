import torch
import torch.nn as nn

class loss_backbone(nn.Module):
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.mse = nn.MSELoss(reduction='none') # We use none to keep it as a vector


    def forward(self,predictions, targets, loss_function_kaggle):
        return loss_function_kaggle(predictions, targets)  # function body of loss_function_kaggleshould be implemented in kaggle
    