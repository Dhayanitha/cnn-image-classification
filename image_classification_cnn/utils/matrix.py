import torch 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import os

def plot_confusion_matrix(model,loader,model_name,device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images,labels in loader:
            images = images.to(device)
            outputs = model(images)
            _,preds= torch.max(outputs,1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels,all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot = True,fmt="d",cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join("image_classification_cnn", "results", f"{model_name}_matrix.png"))
    plt.show()