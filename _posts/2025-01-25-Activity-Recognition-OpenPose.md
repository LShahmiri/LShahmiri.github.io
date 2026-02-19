---
layout: post
title: "Human Activity Recognition – OpenPose Skeleton + 3D CNN Classification"
image: "/posts/HAR3.jpg"
tags: [Deep Learning, OpenPose, 3D-CNN, Streamlit, Pose Estimation, Action Recognition]
---

This project implements a complete **Human Activity Recognition (HAR) system** using:

- **OpenPose COCO-18** for skeleton keypoint extraction  
- A custom **3D ResNet-based CNN** for action classification  
- A fully interactive **Streamlit web application** for real-time inference  

The system processes an input video, extracts 18 body joints per frame, normalizes pose data, and predicts actions such as **boxing, walking, running, jogging, handclapping, and handwaving**.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
- [01. System Design](#design)
- [02. Skeleton Extraction Pipeline](#skeleton)
- [03. 3D CNN Architecture](#architecture)
- [04. Streamlit Application](#app)
- [05. Model Training](#training)
- [06. Code Structure](#code)
- [07. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>
Human activity recognition has growing applications in:

- Elderly fall detection  
- Security systems  
- Sports analytics  
- Smart homes  
- Healthcare monitoring  

Traditional RGB-based models suffer from lighting changes, background noise, and appearance variations.  
**Skeleton-based learning provides a stable representation** for robust action understanding.

---

## Actions <a name="actions"></a>

I developed a full HAR pipeline that:

- Extracts COCO-18 skeleton keypoints from any video  
- Normalizes pose sequences in space + time  
- Uses a **custom 3D ResNet-18 architecture**  
- Performs 5-fold cross-validation  
- Deploys a Streamlit web app for real-time predictions  
- Provides downloadable JSON keypoint files  

---

## Results <a name="results"></a>

- Achieved strong performance across KTH action classes  
- Stable accuracy across folds using dropout + residual blocks  
- Web app runs end-to-end in real time  
- Able to classify:  
  ✔️ boxing  
  ✔️ handclapping  
  ✔️ handwaving  
  ✔️ jogging  
  ✔️ running  
  ✔️ walking  

---

# 01. System Design <a name="design"></a>

Video → OpenPose COCO → JSON Keypoints
→ Preprocessing (Normalize + Sample)
→ 3D CNN Model
→ Softmax Predictions
→ Streamlit UI Output

**Technologies Used**

- OpenPose COCO-18 (Pose Estimation)  
- TensorFlow / Keras (3D CNN)  
- Streamlit (Web UI)  
- NumPy / Pandas / OpenCV  
- Google Colab (GPU training)  

---

# 02. Skeleton Extraction Pipeline <a name="skeleton"></a>

Each frame is processed with OpenPose and produces:

- 18 joints  
- (x, y, confidence) per joint  
- JSON output per frame  

Preprocessing includes:

- Spatial normalization (0–1 scaling)  
- Confidence filtering  
- Temporal alignment to 30 frames  

This produces a final tensor:

(30 frames, 18 joints, 3 features, 1 channel)


Suitable for 3D convolutions.


---

# 03. 3D CNN Architecture <a name="architecture"></a>

The classifier is a **custom 3D ResNet18 variant**, consisting of:

- 3D Conv → BatchNorm → ReLU blocks  
- Residual blocks (64 → 512 filters)  
- MaxPooling along joint axis  
- Dropout (0.3) for generalization  
- Global Average Pooling  
- Softmax classifier  

Key advantages:

- Learns temporal motion  
- Learns spatial joint relationships  
- Lightweight and fast compared to video CNNs  

---

# 04. Streamlit Application <a name="app"></a>

The web app includes:

### **Features**
- Upload any video (mp4/avi/mov)  
- Auto-convert incompatible formats for display  
- Extract skeleton JSON using OpenPose  
- Download JSON keypoints  
- Run 3D CNN classifier  
- Display action + confidence  
- Modern UI with two-column layout  

---

# 05. Model Training <a name="training"></a>

Training approach:

- 5-fold cross validation  
- Adam optimizer (1e-4)  
- Early stopping + ReduceLROnPlateau  
- Custom dataset built from OpenPose JSON files  

Metrics logged:

- Confusion matrix  
- Accuracy / loss curves  
- Fold-level accuracy statistics  

---


# 06. Code Structure <a name="code"></a>

/project
│
├── app.py # Streamlit web UI
├── models/
│ └── best_model_fold_1.keras
├── utils/
│ └── pose_processing.py
│ └── model_utils.py
├── examples/
│ └── sample_input.mp4
│ └── sample_output.json
├── requirements.txt
└── README.md

yaml
Copy code

---

# 07. Discussion <a name="discussion"></a>

This project demonstrates:

- Strong capability of skeleton-based action recognition  
- How CNNs can understand temporal motion patterns  
- How to integrate deep learning with a real-time UI  
- Practical techniques for handling noisy pose estimations  

**Future Work**

- Add fall detection as a new class  
- Support multi-person tracking  
- Deploy as a cloud API  
- Convert model to TensorRT / TFLite for speed  

---

# 🔗 GitHub Repository

👉 [GitHub](https://github.com/LShahmiri/human-activity-recognition-streamlit)

# 🎥 Demo App (Streamlit)

👉 Colab streamlit 

---


