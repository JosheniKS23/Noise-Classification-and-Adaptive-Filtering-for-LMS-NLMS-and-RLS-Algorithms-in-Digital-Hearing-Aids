# Noise-Classification-and-Adaptive-Filtering-for-LMS-NLMS-and-RLS-Algorithms-in-Digital-Hearing-Aids
# AI-Driven Noise Classification and Adaptive Filtering in Digital Hearing Aids

This repository contains the complete implementation for a hybrid framework that combines:

- **Noise classification using UrbanSound8K**
- **Machine Learning & Deep Learning models**
- **Adaptive filtering algorithms (LMS, NLMS, RLS) implemented in Simulink**
- **Performance comparison & signal analysis**
- **Conference publication results**

This work was published in the following IEEE-indexed international conference.

---

## 📜 Published Conference Paper

**AI-Driven Noise Classification and Adaptive Filtering:  
Comparative Performance of LMS, NLMS and RLS Algorithms in Digital Hearing Aids**

*Published in:*  
**Proceedings of the 6th International Conference on Electronics and Sustainable Communication Systems (ICESC-2025)**  
*DVD Part Number:* CFP25V66-DVD  
*ISBN:* 979-8-3315-5502-3  

> This repository includes the full implementation of the experimental setup used in the published paper, including ML/DL models, simulations, and analysis.

---

## 📁 Repository Structure

├── plots/ # All generated ML/DL visualizations

├── tuner_cnn/ # CNN hyperparameter tuning logs & checkpoints

├── tuner_mlp/ # MLP hyperparameter tuning logs

├── Simulink Models/ # LMS, NLMS, RLS adaptive filter simulations

├── UrbanSound8K/ # Dataset folder (user must download separately)

├── IEE_f.py # Complete classification + ensemble pipeline

└── README.md # Project documentation


---

## 🎧 Noise Classification Pipeline (Python + ML/DL)

### 🔹 Features Extracted
- MFCC (40 coefficients)
- Mel-spectrograms (128×128)
- Augmented audio:
- Additive white noise
- Pitch shifting

### 🔹 Models Implemented
- **MLP** (deep fully-connected network)
- **SVM**
- **Random Forest**
- **Convolutional Neural Network (CNN)**
- **Weighted Ensemble Model**

### 🔹 Hyperparameter Optimization
- Keras Tuner (Random Search) for MLP & CNN  
- GridSearchCV for SVM & RF  

### 🔹 Visualizations
All generated inside `plots/`:
- Accuracy & loss curves  
- Confusion matrices  
- ROC curves  
- Model accuracy comparison  
- Ensemble performance graph  

---

## ⚙️ Adaptive Filtering (Simulink)

The electrical-domain portion of the project was developed using **MATLAB/Simulink** to compare:

- **LMS (Least Mean Square)**
- **NLMS (Normalized LMS)**
- **RLS (Recursive Least Square)**

### Analysis Performed:
- Noise power reduction  
- Convergence performance  
- Steady-state error  
- Filter coefficient stability  
- Frequency response evaluation  

The outputs from these simulations form the core of the **adaptive filtering section** of the published paper.

---

## 🧪 Main Script: `IEE_f.py`

This script executes the full ML pipeline:
- Load → augment → extract features → train → tune → evaluate → visualize  
- Saves `.npy` feature files for faster reruns  
- Automatically stores all plots  

---

