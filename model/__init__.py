# Python 3.8.10

import torch.nn as nn


class IDENTITY(nn.Module):
    def __init__(self, pixel_amount):
        super(IDENTITY,self).__init__()

        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # MLP
        self.input_layer = nn.Linear(pixel_amount, pixel_amount / 2) 
        self.hidden_layer = nn.Linear(pixel_amount / 2, pixel_amount / 2)
        self.output_layer = nn.Linear(pixel_amount / 2, pixel_amount)
        
    def forward(self,x):
        out = self.input_layer(x)
        out = self.dropout(out)
        out = self.hidden_layer(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        
        return out