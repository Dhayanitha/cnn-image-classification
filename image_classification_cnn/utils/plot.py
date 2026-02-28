import matplotlib.pyplot as plt
import os

def plot_metrics(train_acc,test_acc,train_loss,val_loss,model_name):

    epochs = range(1,len(train_acc)+1)
    
    plt.figure()
    plt.plot(epochs,train_acc,label="Train Accuracy")
    plt.plot(epochs,test_acc,label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name.upper()} Accuracy")
    plt.legend()
    plt.savefig(os.path.join("image_classification_cnn", "results", f"{model_name}_accuracy.png"))

    plt.figure()
    plt.plot(epochs,train_loss,label = "Train Loss")
    plt.plot(epochs,val_loss,label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss")
    plt.legend()
    plt.savefig(os.path.join("image_classification_cnn", "results", f"{model_name}_loss.png"))

    plt.show()