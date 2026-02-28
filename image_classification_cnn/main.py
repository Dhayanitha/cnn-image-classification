import argparse
from image_classification_cnn.engine.trainer import run_experiment
from image_classification_cnn.models.cnn_v2 import CNN_V2
from image_classification_cnn.models.cnn_v3 import CNN_V3
from image_classification_cnn.models.mlp import MLP
from image_classification_cnn.models.cnn_v1 import CNN
from image_classification_cnn.datasets.dataloader import getdataloader
import image_classification_cnn.config  as config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="mlp",help="Choose model: mlp or cnn")
    args = parser.parse_args()
    MODEL_NAME = args.model.lower()
    trainloader,testloader = getdataloader(config.BATCH_SIZE)
    if MODEL_NAME == "mlp":
        model = MLP()
    elif MODEL_NAME == "cnn_v1":
        model = CNN()
    elif MODEL_NAME == "cnn_v2":
        model = CNN_V2()
    elif MODEL_NAME == "cnn_v3":
        model = CNN_V3()
    else:
        raise ValueError("Model must be 'mlp' or 'cnn_v1' or 'cnn_v2' or 'cnn_v3' " )
    model = model.to(config.DEVICE)

    model = model.to(config.DEVICE)

    trainloader, testloader = getdataloader(config.BATCH_SIZE)

    run_experiment(model, MODEL_NAME, trainloader, testloader, config.DEVICE)


if __name__ == "__main__":
    main()


