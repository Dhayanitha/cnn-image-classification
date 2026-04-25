import torch 
from image_classification_cnn.utils.metrics import calculate_accurracy

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    running_loss =0
    correct = 0
    total = 0
    for images,labels in loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        correct+= calculate_accurracy(outputs,labels)
        total+=labels.size(0)
    return running_loss/len(loader), 100*correct/total

def evaluate(model, loader,criterion, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs,labels)
            total_loss+=loss.item()

            correct += calculate_accurracy(outputs, labels)
            total += labels.size(0)
    
    avg_loss = total_loss/len(loader)
    accuracy = 100 * correct / total

    return avg_loss,accuracy