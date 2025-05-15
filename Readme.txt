# Deep Learning Assignment – Image Classification with PyTorch Lightning

This project explores image classification using deep learning models on the **Imagenette** and **CIFAR-10** datasets. The assignment includes:

- Training a custom Basic CNN
- Training a ResNet-18 model
- Applying regularization (dropout + data augmentation)
- Performing transfer learning from Imagenette to CIFAR-10

All models are implemented using **PyTorch Lightning** for modularity and clarity.

---

## 🔧 Requirements

You can run this project on **Google Colab** with minimal setup. Ensure the following Python packages are installed:

- torch
- torchvision
- pytorch-lightning (or `lightning`)
- torchmetrics

---

##  Models Overview

### 1. Basic CNN
- Built from scratch with Conv2D, ReLU, MaxPool, and FC layers.
- Includes dropout and early stopping.
- Trained on **Imagenette** (grayscale, resized to 64×64).
- **Final Test Accuracy:** ~63%

### 2. ResNet-18
- Standard architecture with input modification for grayscale.
- Trained with early stopping and model checkpointing.
- **Final Test Accuracy:** ~56%

### 3. Regularization (ResNet-18)
- Applied **dropout (p=0.5)** and **data augmentation** (flip, rotate, jitter).
- Improved model robustness on CIFAR-10 dataset.

### 4. Transfer Learning
- Used pretrained ResNet-18 (trained on Imagenette).
- Fine-tuned it on **CIFAR-10**.
- Compared results with model trained from scratch.

---

## 📊 Results Summary

| Model                     | Dataset     | Test Accuracy |
|--------------------------|-------------|----------------|
| Basic CNN                | Imagenette  | ~63.08%        |
| ResNet-18                | Imagenette  | ~56.40%        |
| ResNet-18 + Regularization | CIFAR-10   | Improved vs. base |
| Transfer Learning (Imagenette ➝ CIFAR-10) | CIFAR-10 | Outperformed training from scratch |

---

## 🚀 How to Run

### On Google Colab:
1. Upload the notebook(s) to your Colab environment.
2. Ensure GPU is enabled (Runtime > Change Runtime Type > GPU).
3. Run each notebook cell in order.
4. For transfer learning, make sure to load `imagenette_resnet.ckpt` before fine-tuning.

---



