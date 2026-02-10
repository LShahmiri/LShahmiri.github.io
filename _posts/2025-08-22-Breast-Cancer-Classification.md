---
layout: post
title: "Breast Cancer Histopathology Classification – Reducing Overfitting via MIGT"
image: "/posts/breastcancer.png"
tags: [DeepLearning, CNN, MedicalImaging, Xception, Overfitting, MIGT, Python, TensorFlow]
---

This project investigates how **data selection strategies** influence **overfitting and generalization**
in breast cancer histopathology image classification using deep convolutional neural networks.

Unlike conventional approaches that primarily focus on network architecture or regularization techniques,
this study demonstrates that **dataset partitioning alone can act as an implicit regularizer**,
significantly reducing overfitting under **identical training conditions**.

---

## Project Overview

Breast cancer histopathology classification is a challenging task due to:
- high visual similarity between tissue samples
- strong sensitivity to dataset composition
- limited robustness to biased data splits

As a result, many deep learning models achieve high training accuracy
while suffering from **poor generalization**.

This project shows that **data selection strategy alone** can substantially reduce overfitting,
without modifying the model architecture or training pipeline.

---

## Experimental Design

All experiments were conducted under **strictly identical conditions**:

- Same Xception CNN architecture
- Same training schedule and hyperparameters
- Same data augmentation strategy
- Same RGB input resolution

**Only the dataset selection strategy was changed**, allowing a fair comparison
of its effect on overfitting and generalization.

---

## Dataset & Preprocessing

- **Dataset:** BreaKHis
- **Classes:** Benign / Malignant
- **Input size:** 224 × 224
- **Image formats:** RGB and grayscale variants
- **MIGT subsets:** generated using Mutual Information–based grouping

---

## MIGT vs Random Sampling

### Random Sampling
- High training accuracy
- Large gap between training and validation performance
- Pronounced overfitting

### MIGT (Mutual Information Guided Training)
- High accuracy
- MI-based guided sample selection
- More balanced train/validation/test subsets
- Improved generalization
- Reduced overfitting

---

## Model Architecture

- Backbone: **Xception** (ImageNet pretrained)
- Global Average Pooling
- Dropout (0.3)
- Sigmoid output layer for binary classification

---

## Training Strategy (Two-Stage Training)

### Stage 1 – Head Training
- Backbone frozen
- Train classification head only

### Stage 2 – Fine-Tuning
- Unfreeze last 30 convolutional layers
- Batch Normalization layers remain frozen
- Lower learning rate for training stability

---

## Training Dynamics (Loss & Accuracy)

The following training curves compare two dataset selection strategies
under identical training conditions:
Random sampling, and MIGT on RGB images.

The objective is to illustrate their impact on overfitting and generalization.

### Training vs Validation Loss 

#### Random Sampling

<p align="center"> <img width="846" height="393" alt="31" src="https://github.com/user-attachments/assets/8b51d86e-1084-4f99-b7d6-8013804a34e7" />
 </p>
 


#### MIGT (RGB)

<p align="center"> <img width="846" height="393" alt="11" src="https://github.com/user-attachments/assets/d79545c2-8ae7-4504-b280-923516655081" />
 </p>

Random sampling shows a large and persistent gap between training and validation loss,
indicating severe overfitting.

MIGT (RGB) reduces this gap, suggesting improved regularization via guided data selection.

### Training vs Validation Accuracy

Random Sampling

<p align="center"> <img width="855" height="393" alt="32" src="https://github.com/user-attachments/assets/4baf82ec-42ca-4f58-ad77-1067f7017f8d" />
 </p>


MIGT (RGB)

<p align="center"> <img width="855" height="393" alt="22" src="https://github.com/user-attachments/assets/e05ca9ba-69e5-4e87-b8ad-655c163dc745" />
 </p>


MIGT (RGB) provides the most consistent validation accuracy across epochs,
indicating superior generalization.

### Key Observation:
MIGT-based data selection—particularly on RGB images—substantially reduces overfitting.
This highlights that dataset construction can have a stronger influence on generalization
than model architecture alone.

---

## Discussion

This project demonstrates that:

High accuracy does not necessarily imply good generalization

Overfitting is not solely a model-level problem

Dataset selection can function as an implicit regularizer

Mutual Information provides a principled, data-centric solution



### 🔗 GitHub Repository <a name="GitHub"></a>

👉 [GitHub](https://github.com/LShahmiri/CancerGuard-Breast)

