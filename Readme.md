
# Image Classification using Deep Learning

This repository contains implementations of various image classification models using the **Imagenette** and **CIFAR-10** datasets. The project explores several architectures and training strategies, including a basic CNN, ResNet18, regularization, and transfer learning.

All implementations are done using **PyTorch Lightning** for modularity and scalability.

---

## 📁 Project Structure

- `basic_cnn.py`: Implementation and training of a custom CNN on the Imagenette dataset.
- `resnet_18,_regularization_and_transfer_learning.py`: Implements ResNet18 training on Imagenette, adds regularization, and explores transfer learning to CIFAR-10.

---

## 🔧 Environment Setup

Install dependencies using:

```bash
pip install torch torchvision lightning torchmetrics matplotlib
```

---

## 📚 Tasks and Implementations

### 1. Basic CNN

- **Architecture**:
  - 3 convolutional layers with ReLU and MaxPooling.
  - 2 fully connected layers with dropout (0.3).
  - Grayscale inputs resized to 64x64.
- **Training Details**:
  - Dataset: Imagenette
  - Early stopping and model checkpointing used.
- **Performance**:
  - Training Loss: `0.2346`
  - Validation Loss: `0.3145`
  - Final Test Accuracy: `0.6308`

---

### 2. ResNet18

- **Modifications**:
  - First layer updated to accept grayscale images.
  - Final fully connected layer outputs 10 classes.
- **Training Details**:
  - Dataset: Imagenette
  - Adam optimizer, early stopping, checkpointing.
- **Performance**:
  - Training Loss: `0.2619`
  - Validation Loss: `0.4594`
  - Final Test Accuracy: `0.5640`

---

### 3. Regularization with ResNet18

- **Additions**:
  - Dropout (0.5) before final FC layer.
  - Data Augmentation: Random Flip, Rotation, Color Jitter.
- **Dataset**: CIFAR-10
- **Performance**:
  - Training Loss: `0.1813`
  - Validation Loss: `0.0130`
  - Final Test Accuracy: `0.8206`

---

### 4. Transfer Learning

- **Strategy**:
  - Fine-tune ResNet18 pre-trained on Imagenette.
  - Compare with a ResNet18 trained from scratch on CIFAR-10.
- **Results**:

| Metric                  | Fine-Tuned Model | Trained from Scratch |
|------------------------|------------------|-----------------------|
| Final Epoch            | 12               | 11                    |
| Train Accuracy (epoch) | 0.988            | 0.983                 |
| Validation Accuracy    | 0.834            | 0.786                 |
| Train Loss (epoch)     | 0.0508           | 0.0521                |
| Validation Loss        | 0.650            | 0.964                 |

Transfer learning led to improved generalization and faster convergence.

---

## 📊 Visualization

All training histories were plotted using Matplotlib. You can use the `plot_losses()` function in each script to view training and validation loss trends.

---

## 📌 Conclusion

This project demonstrates the effectiveness of:
- Custom CNNs for baseline performance.
- Deeper architectures like ResNet18 for enhanced learning.
- Regularization (dropout, augmentation) for generalization.
- Transfer learning for leveraging pre-trained knowledge.

---

## 🚀 How to Run

1. Clone the repository.
2. Open scripts in Google Colab (GPU preferred).
3. Run each cell step-by-step to train and evaluate models.

---



---
