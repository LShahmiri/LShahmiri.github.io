---
layout: post
title: "Breast Cancer Histopathology Classification – Reducing Overfitting via MIGT"
image: "/posts/breastcancer.png"
tags: [DeepLearning, CNN, MedicalImaging, Xception, Overfitting, MIGT, Python, TensorFlow]
---

This project investigates how data selection strategies influence overfitting and generalization
in breast cancer histopathology image classification using deep convolutional neural networks.

Unlike conventional approaches that focus solely on model architecture, this study demonstrates
that dataset partitioning alone can act as an implicit regularizer, significantly reducing overfitting
under identical training conditions.

---

# Project Overview

Breast cancer histopathology classification is a challenging task due to high visual similarity
between samples and strong sensitivity to dataset composition.
Many deep learning models achieve high accuracy while suffering from severe overfitting.

This project shows that data selection strategy alone can significantly reduce overfitting
without modifying the model architecture.

---

# Experimental Design

All experiments were conducted under strictly identical conditions:
- Same Xception CNN architecture
- Same training schedule and hyperparameters
- Same RGB images

Only the dataset selection strategy was changed.

---

# Dataset & Preprocessing

Dataset: BreaKHis  
Classes: Benign / Malignant  
Input size: 224 × 224  
RGB and grayscale variants evaluated  
MIGT subsets generated using Mutual Information  

---

# MIGT vs Random Sampling

Random Sampling  
- High accuracy  
- Severe overfitting  

MIGT (Mutual Information Guided Training)  
- MI-based sample selection  
- Better generalization  
- Reduced overfitting  

---

# Model Architecture

Backbone: Xception (ImageNet pretrained)  
Global Average Pooling  
Dropout (0.3)  
Sigmoid output layer  

---

# Training Strategy (Two-Stage Training)

Stage 1 – Head Training  
- Base model frozen  
- Train classifier head  

Stage 2 – Fine-Tuning  
- Unfreeze last 30 convolutional layers  
- Batch Normalization layers remain frozen  
- Lower learning rate for stability  

---

# Configuration (YAML-style)

dataset:
  base_dir: data/MIGT_DATASET_RGB
  train_dir: train
  val_dir: val
  test_dir: test

data:
  img_size: [224, 224]
  batch_size: 32
  seed: 42

training:
  epochs: 100
  patience: 6
  head_lr: 0.001

fine_tuning:
  enabled: true
  unfreeze_last_layers: 30
  lr: 0.00001
  epochs: 20

output:
  save_dir: outputs/migt_rgb

---

# Core Python Logic (Simplified)

base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.xception.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

for layer in base_model.layers[-30:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

---
# Training Dynamics (Loss & Accuracy)

The following training curves compare **three dataset selection strategies**
under **identical training conditions**:
**Random sampling**, **MIGT on RGB images**, and **MIGT on grayscale images**.
The goal is to highlight their impact on overfitting and generalization.

---

### Training vs Validation Loss
**Random Sampling** 

<p align="center">
  <img width="846" height="393"
       alt="rand-loss"
       src="https://github.com/user-attachments/assets/aa90f1de-2b67-428f-9c15-4f44faae17ed" />
</p>

**MIGT (Grayscale)**

<p align="center">
  <img width="846" height="393"
       alt="gray-migt-loss"
       src="https://github.com/user-attachments/assets/f21c1966-02cd-49c7-8026-773f5c17a23d" />
</p>

**MIGT (RGB)** 

<p align="center">
  <img width="846" height="393"
       alt="rgb-migt-loss"
       src="https://github.com/user-attachments/assets/febc284a-c62f-4ac1-99e1-0859c5952780" />
</p>

- **Random Sampling** exhibits a large gap between training and validation loss,
  indicating severe overfitting.
- **MIGT (Grayscale)** reduces this gap, suggesting improved regularization
  through guided sample selection.
- **MIGT (RGB)** shows the smallest loss gap and the most stable convergence,
  demonstrating the strongest reduction in overfitting.

---

### Training vs Validation Accuracy
**Random Sampling**

<p align="center">
  <img width="846" height="393"
       alt="rand-acc"
       src="https://github.com/user-attachments/assets/309faba0-03b5-4503-afe0-e8b20d87616e" />
</p>

**MIGT (Grayscale)**

<p align="center">
  <img width="846" height="393"
       alt="gray-migt-acc"
       src="https://github.com/user-attachments/assets/5517b7aa-c395-4d70-8609-8f00525d7e29" />
</p>

**MIGT (RGB)**

<p align="center">
  <img width="846" height="393"
       alt="rgb-migt-acc"
       src="https://github.com/user-attachments/assets/fde851b0-fa74-424e-ba25-58e686371c5a" />
</p>

- **Random Sampling** achieves high training accuracy but suffers from unstable
  validation performance, reflecting poor generalization.
- **MIGT (Grayscale)** improves validation stability compared to random sampling,
  though some fluctuation remains.
- **MIGT (RGB)** provides the most consistent validation accuracy across epochs,
  indicating superior generalization under the same model and hyperparameters.

---

**Key Observation:**  
While all three approaches may reach comparable peak accuracy, **MIGT-based data
selection—particularly on RGB images—substantially reduces overfitting**, showing
that dataset construction has a stronger influence on generalization than model
architecture alone.

---

# Discussion

This project demonstrates that:
- High accuracy does not imply good generalization
- Overfitting is not solely a model-level problem
- Dataset selection can act as an implicit regularizer
- Mutual Information provides a principled data-centric solution
- RGB-based MIGT reduces overfitting more effectively than grayscale MI selection

This work aligns with my PhD research on robust deep learning technique for medical imaging.

---
### 🔗 GitHub Repository  
👉 [GitHub](https://github.com/LShahmiri/CancerGuard-Breast)
