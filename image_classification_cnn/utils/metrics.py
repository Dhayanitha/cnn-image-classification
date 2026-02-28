import torch 

def calculate_accurracy(outputs,labels):
    _,predicted = torch.max(outputs,1)
    correct = (predicted==labels).sum().item()
    return correct