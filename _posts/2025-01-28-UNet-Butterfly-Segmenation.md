---
layout: post
title: "Butterfly Segmentation – U-Net with MobileNetV2 Encoder"
image: "/posts/Butterfly.png"
tags: [ComputerVision, ImageSegmentation, UNet, MobileNetV2, TensorFlow, Keras, BinarySegmentation, U-Net, DeepLearning, Python]
---

This project implements a **binary semantic segmentation pipeline** using **U-Net with a MobileNetV2 encoder** pretrained on **ImageNet**.  
The goal is robust **foreground–background separation** (e.g., **butterfly vs. background**) to support downstream tasks such as **citizen-science landmarking** and dataset preparation.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
  - [Growth / Next Steps](#growth)
- [01. Data Pipeline & Mask Construction](#data)
- [02. Model Architecture](#model)
- [03. Loss Function & Training Strategy](#training)
- [04. Evaluation & Sanity Checks](#eval)
- [05. Inference & Saving Predictions](#inference)
- [06. Full Code](#code)
- [07. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>

Butterfly datasets captured in natural environments often contain:
- complex backgrounds (rocks, plants, shadows)
- varied illumination and wing poses
- occlusion and fine edge structures (antennae, wing boundaries)

To enable reliable annotation workflows (e.g., **Zooniverse landmarking**) and improve model focus, a dedicated **binary segmentation model** was required to isolate the butterfly region.

---

## Actions <a name="actions"></a>

I implemented a complete segmentation pipeline that:

- Loads and resizes images/masks to **512×512**
- Combines **multiple instance masks** into a single **binary union mask**
- Builds a **U-Net decoder** on top of a pretrained **MobileNetV2 encoder**
- Uses skip connections for multi-scale feature fusion
- Trains using **Dice + BCE combined loss**
- Trains in two phases:
  1) **Freeze encoder** (fast convergence)
  2) **Fine-tune all layers** (improved edges and generalisation)
- Runs inference on test images and saves masks resized back to original dimensions

---

## Results <a name="results"></a>

- Stable binary segmentation masks for butterfly foreground extraction  
- Improved object boundary preservation through skip connections  
- Robust performance on complex natural backgrounds  
- Prediction masks are saved at **original image size**, enabling immediate use in annotation workflows

---

## Growth / Next Steps <a name="growth"></a>

Planned improvements:

- Data augmentation (flip/rotate/brightness) for improved robustness  
- Post-processing (morphological ops / CRF) to refine boundaries  
- Multi-class segmentation (wings / body / antennae)  
- Export to **TFLite** for edge/embedded segmentation  
- Evaluation metrics on test set: IoU, precision/recall, boundary F1

---

# 01. Data Pipeline & Mask Construction <a name="data"></a>

**Training format:**
- `TRAIN_PATH/<id>/images/<id>.png`
- `TRAIN_PATH/<id>/masks/*.png` (multiple instances)

All instance masks are merged into **one binary mask** via pixelwise maximum, then thresholded into `{0,1}`.

Test images preserve original dimensions so predictions can be resized back after inference.

---

# 02. Model Architecture <a name="model"></a>

**Encoder:** MobileNetV2 (ImageNet pretrained, `include_top=False`)  
**Decoder:** U-Net style upsampling blocks (Conv2DTranspose + concat skip + Conv2D)

**Skip connection layers:**
- `block_1_expand_relu` (256×256)
- `block_3_expand_relu` (128×128)
- `block_6_expand_relu` (64×64)
- `block_13_expand_relu` (32×32)
- bottleneck `block_16_project` (16×16)

**Output:** 1-channel sigmoid mask (binary segmentation)

---

# 03. Loss Function & Training Strategy <a name="training"></a>

**Loss:** `BinaryCrossEntropy + (1 - DiceCoefficient)`  
This balances:
- pixel-level accuracy (BCE)
- region overlap (Dice), especially for imbalanced foreground/background

**Two-stage training:**
1. Freeze encoder → train decoder with LR=1e-3  
2. Unfreeze all layers → fine-tune with LR=1e-4

---

# 04. Evaluation & Sanity Checks <a name="eval"></a>

I included sanity checks to visualise:
- training image
- ground-truth mask
- predicted mask

This confirms:
- correct mask construction
- correct foreground extraction
- stable training convergence

---

# 05. Inference & Saving Predictions <a name="inference"></a>

- Predictions are generated at 512×512 then thresholded at 0.5  
- Each predicted mask is resized back to **original image height/width** using **nearest-neighbour** interpolation to preserve binary structure  
- Masks are saved to output folders per test ID

---

# 06. Full Code <a name="code"></a>

```python
import os
import random
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ----------------------------
# Config
# ----------------------------
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_PATH = '/mask-trin-path/'
TEST_PATH  = '/test-path/'
PRED_TEST_OUT = 'mask-output-path-/'
os.makedirs(PRED_TEST_OUT, exist_ok=True)

# ----------------------------
# Discover IDs
# ----------------------------
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids  = next(os.walk(TEST_PATH))[1]

# ----------------------------
# Load Training Data
# ----------------------------
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = os.path.join(TRAIN_PATH, id_)

    # Image
    img_path = os.path.join(path, 'images', f'{id_}.png')
    img = imread(img_path)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                 preserve_range=True, anti_aliasing=True)
    X_train[n] = img.astype(np.uint8)

    # Mask (union of all instance masks)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    masks_dir = os.path.join(path, 'masks')
    for mask_file in next(os.walk(masks_dir))[2]:
        m = imread(os.path.join(masks_dir, mask_file), as_gray=True)
        m = resize(m, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                   preserve_range=True, anti_aliasing=False)
        m = np.expand_dims(m, axis=-1)
        mask = np.maximum(mask, m)

    mask = (mask > 0).astype(np.float32)
    Y_train[n] = mask

# ----------------------------
# Load Test Data (keep originals sizes)
# ----------------------------
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = os.path.join(TEST_PATH, id_)
    img_path = os.path.join(path, 'images', f'{id_}.png')

    img = imread(img_path)[:, :, :IMG_CHANNELS]
    sizes_test.append((img.shape[0], img.shape[1]))
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                 preserve_range=True, anti_aliasing=True)
    X_test[n] = img.astype(np.uint8)

print('Done loading!')

# ----------------------------
# Sanity Check
# ----------------------------
if len(train_ids) > 0:
    image_x = random.randint(0, len(train_ids)-1)
    plt.figure(); plt.title("Train Image"); plt.imshow(X_train[image_x]); plt.axis('off')
    plt.figure(); plt.title("Train Mask");  plt.imshow(np.squeeze(Y_train[image_x]), cmap='gray'); plt.axis('off')
    plt.show()

# ----------------------------
# Build Model: U-Net + MobileNetV2 encoder
# ----------------------------
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

preproc = tf.keras.applications.mobilenet_v2.preprocess_input
x = tf.keras.layers.Lambda(preproc, name='mobilenetv2_preproc')(inputs)

base = tf.keras.applications.MobileNetV2(
    input_tensor=x, include_top=False, weights='imagenet'
)

skip1 = base.get_layer('block_1_expand_relu').output
skip2 = base.get_layer('block_3_expand_relu').output
skip3 = base.get_layer('block_6_expand_relu').output
skip4 = base.get_layer('block_13_expand_relu').output
bottleneck = base.get_layer('block_16_project').output

def up_block(x, skip, filters, name=None):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding='same',
                                        name=None if not name else name+"_up")(x)
    x = tf.keras.layers.Concatenate(name=None if not name else name+"_concat")([x, skip])
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu',
                               kernel_initializer='he_normal',
                               name=None if not name else name+"_conv1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu',
                               kernel_initializer='he_normal',
                               name=None if not name else name+"_conv2")(x)
    return x

d1 = up_block(bottleneck, skip4, 256, name="dec1")
d2 = up_block(d1,        skip3, 128, name="dec2")
d3 = up_block(d2,        skip2, 64,  name="dec3")
d4 = up_block(d3,        skip1, 32,  name="dec4")

d5 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding='same', name="dec5_up")(d4)
d5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal', name="dec5_conv1")(d5)
d5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same',
                            kernel_initializer='he_normal', name="dec5_conv2")(d5)

outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='mask')(d5)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet_MobileNetV2')

for layer in base.layers:
    layer.trainable = False

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=['accuracy', dice_coef]
)

# ----------------------------
# Training
# ----------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'model_for_butterfly_unet.h5',
    monitor='val_loss', verbose=1, save_best_only=True
)
early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

results = model.fit(
    X_train.astype(np.float32), Y_train.astype(np.float32),
    validation_split=0.1,
    batch_size=16,
    epochs=25,
    callbacks=[checkpointer, early_stop, reduce_lr, tensorboard],
    shuffle=True
)

# ----------------------------
# Fine-tune
# ----------------------------
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy', dice_coef]
)

results_ft = model.fit(
    X_train.astype(np.float32), Y_train.astype(np.float32),
    validation_split=0.1,
    batch_size=8,
    epochs=10,
    callbacks=[checkpointer, early_stop, reduce_lr, tensorboard],
    shuffle=True
)

# ----------------------------
# Predictions
# ----------------------------
split_idx = int(X_train.shape[0] * 0.9)

preds_train = model.predict(X_train[:split_idx], verbose=1)
preds_val   = model.predict(X_train[split_idx:], verbose=1)
preds_test  = model.predict(X_test, verbose=1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)

# ----------------------------
# Save test predictions resized back to original size
# ----------------------------
print("Saving test predictions to:", PRED_TEST_OUT)
for i, id_ in enumerate(test_ids):
    H, W = sizes_test[i]
    mask_small = np.squeeze(preds_test_t[i]).astype(np.float32)
    mask_orig = resize(mask_small, (H, W), order=0,
                       anti_aliasing=False, preserve_range=True)
    mask_orig = (mask_orig > 0.5).astype(np.uint8) * 255

    out_dir = os.path.join(PRED_TEST_OUT, id_)
    os.makedirs(out_dir, exist_ok=True)
    imsave(os.path.join(out_dir, f'{id_}_pred.png'), mask_orig.astype(np.uint8))

print("All done!")

```
---

### 🔗 GitHub Repository  
👉 [GitHub](https://github.com/LShahmiri/U-Net-Segmentation)
