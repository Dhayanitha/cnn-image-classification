import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def getdataloader(batchsize):
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    trainloader = DataLoader(trainset,batch_size=batchsize,shuffle=True)
    testLoader = DataLoader(testset,batch_size=batchsize,shuffle=True)

    return trainloader,testLoader
