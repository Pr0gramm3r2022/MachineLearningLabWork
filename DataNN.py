import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#defines what device will be used for training
#reformatted from original format
device = 0

if torch.accelerator.is_available() is True:
    device = torch.accelerator.current_accelerator().type 
else:
    device = "cpu"

#should ouput "Using cuda device" or "Using cpu"

print(f"Using {device} device")


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
'''what is the significance of these numbers? are these the 
dimensions of the different kinds of layers that will be initialized/made?'''

    

model = NeuralNet().to(device)
print(model)

def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        #need to go over what logits is/are
        return logits

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probability = nn.Softmax(dim=1)(logits)
y_pred = pred_probability.argmax(1)
#is y_pred that actual probability?
print(f"Predicted class: {y_pred}")

