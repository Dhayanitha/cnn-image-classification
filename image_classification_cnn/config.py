import torch 
DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001