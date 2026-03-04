# CNN vs MLP for Image Classification (PyTorch)

This project implements and compares multiple Convolutional Neural Network (CNN) architectures with a Multi-Layer Perceptron (MLP) baseline for image classification using PyTorch.

The objective is to demonstrate the architectural advantages of CNNs over fully connected networks when working with spatial image data, while building a modular and production-style deep learning training pipeline.

## 1. Problem Statement

Fully connected neural networks (MLPs) treat images as flattened vectors, ignoring spatial structure. Convolutional Neural Networks (CNNs) preserve spatial locality through convolution operations, making them more suitable for image tasks.

This project experimentally compares MLP and CNN architectures on image classification to quantify:

* Performance differences
* Generalization ability
* Parameter efficiency
* Training stability

## 2. Dataset

Dataset: CIFAR-10

* 60,000 RGB images
* 50,000 training images
* 10,000 test images
* 10 object classes
* Image size: 32 × 32

Preprocessing:

* Tensor conversion
* Normalization using dataset mean and standard deviation

## 3. Project Architecture

The project follows a clean, modular structure inspired by production ML systems.

```
image_classification_cnn/
│
├── models/
│   ├── mlp.py
│   ├── cnn_v1.py
│   ├── cnn_v2.py
│   └── cnn_v3.py
│
├── datasets/
│   └── dataloader.py
│
├── engine/
│   └── trainer.py
│
├── utils/
│   ├── metrics.py
│   ├── matrix.py
│   └── plot.py
│
├── config.py
├── train.py
└── main.py
```

Design Principles:

* Separation of concerns (model / data / training / utilities)
* Config-driven experimentation
* Validation-based checkpoint saving
* Reproducible training setup
* Easily extensible architecture


## 4. Models Implemented

MLP Baseline

* Fully connected network
* Flattens 32×32×3 image into vector
* Serves as architectural comparison baseline

CNN v1

* Basic convolutional architecture
* Conv → ReLU → Pool blocks
* Fully connected classifier

CNN v2

* Deeper architecture
* Increased channel capacity
* Improved feature extraction

CNN v3

* Enhanced depth
* Batch Normalization
* Dropout regularization
* Improved generalization performance

## 5. Training Features

* CrossEntropyLoss
* Adam optimizer
* Learning rate scheduling using StepLR
* Model checkpointing based on validation accuracy
* Accuracy and loss tracking
* Confusion matrix evaluation
* TensorBoard logging
* Reproducible seed configuration

## 6. Results

Model Comparison:
```
Model        | Validation Accuracy
MLP          | ~60–65%
CNN v1       | ~75–78%
CNN v2       | ~82–84%
CNN v3       | ~85%+
```

## 7. How to Run

Train a specific model:

python -m image_classification_cnn.main --model cnn_v3

Run TensorBoard:

tensorboard --logdir runs

## 8. Environment Setup

Requirements:

* Python 3.10+
* PyTorch
* torchvision
* matplotlib
* scikit-learn
* tensorboard

Install dependencies:

pip install -r requirements.txt

## 9. Future Improvements

* Implement ResNet architecture
* Add data augmentation
* Hyperparameter search (Optuna / grid search)
* Mixed precision training
* Model deployment using FastAPI
* Docker containerization

## Author
Dhayanitha H
