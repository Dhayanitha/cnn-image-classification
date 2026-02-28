import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from image_classification_cnn.utils.plot import plot_metrics
from image_classification_cnn.train import train_one_epoch, evaluate
from image_classification_cnn.utils.matrix import plot_confusion_matrix
import image_classification_cnn.config as config


def run_experiment(model, model_name, trainloader, testloader, device):
    

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    RUN_DIR = os.path.join(BASE_DIR,"run")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    val_loss_history =[]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
    
    writer = SummaryWriter(log_dir=RUN_DIR)
    best_val_loss = float("inf")
    model.to(device)

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )

        val_loss,test_acc = evaluate(model, testloader, criterion,device)
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)


        if val_loss < best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(),os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth"))

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Acc: {test_acc:.2f}%"
        )

    writer.close()

    plot_confusion_matrix(model, testloader,model_name,device)
    plot_metrics(train_acc_history, test_acc_history,train_loss_history,val_loss_history, model_name)