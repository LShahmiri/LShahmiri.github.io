---
layout: post
title: "Breast Cancer Histopathology Classification – Reducing Overfitting via MIGT"
image: "/posts/breast_cancer_migt.png"
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

# Discussion

This project demonstrates that:
- High accuracy does not imply good generalization
- Overfitting is not solely a model-level problem
- Dataset selection can act as an implicit regularizer
- Mutual Information provides a principled data-centric solution
- RGB-based MIGT reduces overfitting more effectively than grayscale MI selection

This work aligns with ongoing PhD research on robust and explainable deep learning
for medical imaging.

---

GitHub Repository: add your link here  

Disclaimer:  
This project is for research and educational purposes only and is not intended for clinical use.
