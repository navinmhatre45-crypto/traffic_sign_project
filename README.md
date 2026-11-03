# 🚦 Conditional GAN-Driven Traffic Sign Recognition System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io/)

A complete deep learning framework that integrates a **Conditional Generative Adversarial Network (cGAN)** with a **Deep Convolutional Neural Network (CNN)** on the German Traffic Sign Recognition Benchmark (GTSRB).

---

## 📖 Executive Summary & Academic Framing

### The Core Problem: Extreme Class Imbalance in Autonomous Driving
In standard traffic sign datasets such as GTSRB, real-world data collection exhibits a severe **long-tail distribution**:
- High-frequency classes (e.g., *Speed limit 30 km/h*, *Speed limit 50 km/h*) have **1,500 training images**.
- Critical minority safety classes (e.g., *Speed limit 20 km/h*, *Dangerous curve left*, *End of no passing*) have only **150 images** — a **10:1 imbalance ratio**.

Standard supervised classifiers suffer from biased decision boundaries that fail on life-critical rare signs despite showing high overall accuracy. Standard affine transformations (rotation, scaling) cannot synthesize novel illumination, textures, or edge conditions.

### The GAN Solution
This project implements a **Conditional DCGAN (cDCGAN)** that learns the class-conditional probability distribution $p(x|y)$ of traffic signs:
1. **Generative Modeling:** Synthesizes realistic, class-conditioned traffic sign variations on demand.
2. **Synthetic Data Augmentation:** Balances the underrepresented classes in the training pipeline.
3. **Latent Manifold Representation:** Demonstrates smooth semantic transitions across the latent space.

---

## 🏗️ System Architecture

```
                       [ Latent Noise Vector z ~ N(0, I) ]
                                      +
                         [ Class Label y in [0, 42] ]
                                      |
                                      v
                     +----------------------------------+
                     |    Conditional Generator G(z, y) |
                     |   (Transposed Convolutions)      |
                     +----------------------------------+
                                      |
                                      v
                           Synthesized Sign (32x32x3)
                                      |
        +-----------------------------+----------------------------+
        |                                                          |
        v                                                          v
+-------------------------------+                         +-----------------------------+
| Conditional Discriminator     |                         | Downstream TrafficSignCNN   |
| D(x, y) [Minimax Training]    |                         | (Classification Validation) |
+-------------------------------+                         +-----------------------------+
```

### 1. Conditional Generator ($G$)
- **Input:** Latent vector $z \in \mathbb{R}^{100}$ concatenated with a 32-dim learnable class embedding.
- **Layers:** Dense projection $\to 256 \times 4 \times 4 \to$ three `ConvTranspose2d` blocks with `BatchNorm2d` and `ReLU`.
- **Output:** $3 \times 32 \times 32$ image with `Tanh` activation (pixel values in $[-1, 1]$).

### 2. Conditional Discriminator ($D$)
- **Input:** $3 \times 32 \times 32$ image concatenated with a spatial class embedding mask ($1 \times 32 \times 32$) $\to$ 4 input channels.
- **Layers:** Three `Conv2d` downsampling blocks with `LeakyReLU(0.2)` and `BatchNorm2d`.
- **Output:** Single scalar probability via `Sigmoid()` indicating real vs. fake.

### 3. Traffic Sign Classifier (CNN)
- 6 convolutional blocks with Batch Normalization, Max Pooling, and Dropout.
- 2 fully connected classification layers mapping to 43 output classes.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `gan.py` | Architecture of `ConditionalGenerator`, `ConditionalDiscriminator`, and sampling utilities. |
| `train_gan.py` | Full training script for the Conditional GAN with visual grid checkpoints. |
| `generate_signs.py` | CLI tool to synthesize signs, generate multi-class galleries, and test classifier accuracy. |
| `train_augmented.py` | Fine-tunes/evaluates the classifier using real + GAN-augmented synthetic data. |
| `app.py` | Interactive Streamlit web app: live classifier, GAN generation studio, and teacher defense slides. |
| `common.py` | Shared dataset transforms, class names, device detection (`mps`/`cuda`/`cpu`), and CNN model. |
| `train.py` | Baseline CNN training script on GTSRB dataset. |
| `predict.py` | Single-image CLI prediction utility. |

---

## 🚀 How to Run

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
```

### 2. Launch the Interactive Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app features:
1. **Traffic Sign Recognition:** Upload any image or test sample signs.
2. **cGAN Synthetic Sign Studio:** Select any of the 43 classes and generate synthetic variations live!
3. **Research & Defense Deck:** Review the architecture and defense Q&A.

### 3. Generate Signs from CLI
```bash
# Generate 16 synthetic Stop signs (Class 14) and verify with CNN
python3 generate_signs.py --class_id 14 --num_samples 16 --output stop_signs.png --classify

# Generate a gallery of all 43 classes
python3 generate_signs.py --gallery --output cgan_gallery.png

# Interpolate between Stop sign (14) and No Entry (17)
python3 generate_signs.py --interpolate --class_id 14 --class_b 17 --output morph.png
```

### 4. Train the Conditional GAN
```bash
python3 train_gan.py --epochs 10 --batch_size 128
```

### 5. Evaluate GAN-Augmented Training
```bash
python3 train_augmented.py --samples_per_minority_class 250 --epochs 5
```

---

## 🎓 How to Present this Project to Your Teacher

### 30-Second Elevator Pitch
> *"Our project tackles one of the greatest safety hazards in autonomous driving: **extreme class imbalance in road sign datasets**. While common speed signs have thousands of samples, critical warning signs have as few as 150 images. Rather than simply training a passive CNN classifier, our project is built around a **Conditional Generative Adversarial Network (cGAN)** that learns the generative distribution of traffic signs. We use the GAN to synthetically generate minority sign classes, balance the training distribution, and systematically improve downstream recognition safety on rare road signs."*

### Key Viva / Defense Questions:
1. **Why is this project GAN-based?**
   - The primary research contribution is generative: synthesizing realistic traffic signs under class constraints and using generative data balancing to fix long-tail failure modes.
2. **Why Conditional GAN instead of vanilla GAN?**
   - Vanilla GANs generate random images from arbitrary classes without control. A Conditional GAN enforces label conditioning $p(x|y)$, allowing targeted synthesis of specific rare classes.
3. **How do you prevent Mode Collapse?**
   - We utilized DCGAN architectural guidelines (Radford et al.): batch normalization, one-sided label smoothing ($0.9$ for real targets), and Adam optimizer with $\beta_1=0.5$.

