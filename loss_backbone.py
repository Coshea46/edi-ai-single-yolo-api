import torch
import torch.nn as nn

class loss_backbone(nn.Module):
    def __init__(self, confidence_weight, coord_weight):
        super().__init__()

        self.confidence_weight = confidence_weight
        self.coord_weight = coord_weight


    def forward(self,predictions, targets, loss_function_kaggle):
        return loss_function_kaggle(predictions, targets, self.confidence_weight, self.coord_weight)  # function body of loss_function_kaggleshould be implemented in kaggle
    