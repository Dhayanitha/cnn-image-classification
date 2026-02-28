# CNN vs MLP for Image Classification (PyTorch)

This project implements and compares multiple Convolutional Neural Network (CNN) architectures with a Multi-Layer Perceptron (MLP) baseline for image classification using PyTorch.

The objective is to demonstrate the architectural advantages of CNNs over fully connected networks for spatial data and to build a modular, extensible deep learning training pipeline.

## 1. Project Objective

- Compare MLP and CNN performance on image classification.
- Implement multiple CNN variants with increasing complexity.
- Build a structured and reusable training pipeline.
- Apply training improvements such as learning rate scheduling and model checkpointing.
- Evaluate models using accuracy metrics and confusion matrix analysis.

## 2. Architecture Overview

The project follows a modular design:

- **Models**: Separate files for each architecture (MLP, CNN v1, v2, v3).
- **Datasets**: Centralized data loading logic.
- **Engine**: Training and evaluation logic.
- **Utilities**: Metrics, plotting, and confusion matrix.
- **Configuration**: Hyperparameters stored in `config.py`.

This separation ensures maintainability, clarity, and scalability.

## 3. Project Structure

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

## 4. Technical Features

- Modular training pipeline
- Multiple CNN implementations
- MLP baseline comparison
- Learning rate scheduler (StepLR)
- Validation-based model checkpointing
- TensorBoard logging
- Confusion matrix visualization
- Accuracy and loss tracking across epochs
- Command-line model selection using argparse

## 5. Training Details

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Scheduler: StepLR
- Checkpointing: Best model saved based on validation performance
- Logging: TensorBoard

## 6. Results

- CNN architectures significantly outperform the MLP baseline.
- Deeper CNN variants achieve ~85%+ validation accuracy.
- Scheduler improves training stability and convergence.

This confirms the importance of spatial feature extraction for image-based tasks.

## 7. How to Run

### Train a Model
python -m image_classification_cnn.main --model cnn_v3

### Launch TensorBoard
tensorboard --logdir runs

## 8. Future Improvements
- Data augmentation
- Early stopping
- Advanced schedulers (CosineAnnealing, ReduceLROnPlateau)
- Hyperparameter tuning
- Transfer learning experiments

## Author

Dhayanitha
