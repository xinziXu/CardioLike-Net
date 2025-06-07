
# CardioLike-Net

**CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications**

This repository contains the official implementation of the CardioLike-Net algorithm proposed in our paper. It is designed for efficient and accurate arrhythmia classification on wearable devices, leveraging quantization-aware training and a lightweight architecture.

---

## 📝 Paper

If you find this work useful in your research, please cite our paper:

> Xinzi Xu, *CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications*, 2025.  

---

## 🧩 Features

- 🔋 **Edge-end inference** with low resource usage  
- 🧠 **Inter-patient classification**  
- ⚙️ **Quantization-aware training (QAT)** for hardware deployment   

---

## 📦 Environment Setup

```bash
# Ensure you have conda installed, then run:
conda env create -f env.yml
conda activate cardiolike-net
```

---

## 🚀 Inference

```bash
# Run the following to perform inference with the pretrained model
python3 main.py
```

This will:

- Load the trained model  
- Run inference on the dataset  
- Output classification results  

---



## 🔒 License

```
Copyright (c) 2025 Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code for non-commercial research purposes, 
provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission.

---
