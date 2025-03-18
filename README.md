# Variational Autoencoders on MNIST: Experimental Results

This repository contains experiments on various variational autoencoder-based architectures, evaluating their reconstruction accuracy and generative quality. Below is a summary of the models tested, the metrics used, and the tradeoff analysis.

---

## Overview

All models were trained on the MNIST dataset with the following hyperparameters:

- **Batch Size:** 256  
- **Learning Rate:** 0.001  
- **Epochs:** 10  
- **Latent Dimension (\(z\))**: 20 (unless specified)  

We evaluate the following autoencoder architectures:

### 1. **Variational Autoencoders (VAEs)**
   - `vae_std_z20`: A standard VAE ([PyTorch implementation](https://github.com/pytorch/examples/blob/main/vae/main.py)) with \(z=20\).
   - `vae_std_z20_res`: A VAE with residual connections added to the encoder.
   - `vae_std_z40`: A larger latent space version (\(z=40\)).
   - `vae_conv`: A convolutional VAE.

### 2. **\(\beta\)-VAEs**
   - `beta_vae_4`: \(\beta=4\)  
   - `beta_vae_2`: \(\beta=2\)  
   - `beta_vae_0.5`: \(\beta=0.5\)  

### 3. **Other Autoencoder Variants**
   - **WAE** (Wasserstein Autoencoder): Focuses on **distribution matching** instead of KL divergence.  
   - **GM-VAE** (Gaussian Mixture VAE): Uses a mixture of Gaussian priors to model a more flexible latent space.

---

## Metrics

We evaluate each model using two key metrics:

1. **FID (Fréchet Inception Distance)**  
   - Measures the quality of generated images by comparing their feature distributions with real images.  
   - Lower is better.

2. **MSE (Mean Squared Error)**  
   - Measures reconstruction accuracy between input and output images.  
   - Lower is better.

---

## Results
### MSE Scores (Lower is Better)

| Model             | MSE Score   |
|-------------------|-------------|
| vae_std_z20       | 0.0144      |
| vae_std_z20_res   | 0.0169      |
| vae_std_z40       | 0.0139  |
| vae_conv_10       | 0.0236      |
| beta_vae_4        | 0.0272      |
| beta_vae_2        | 0.0189      |
| beta_vae_0.5      | 0.0116      |
| WAE               | 0.0092      |
| GM-VAE            | **0.0090**  |

### FID Scores (Lower is Better)

| Model             | FID Score               |
|-------------------|-------------------------|
| vae_std_z20       | 64.45                    |
| vae_std_z20_res   | **41.26**        |
| vae_std_z40       | 63.33                    |
| vae_conv_10       | 50.13                    |
| beta_vae_4        | 72.12                    |
| beta_vae_2        | 64.51                    |
| beta_vae_0.5      | 71.99                    |
| WAE               | 176.34                   |
| GM-VAE            | 247.84                   |


---

## Tradeoff Analysis

### **1. Standard VAEs (Effect of Residuals & Latent Dim)**
- **`vae_std_z20_res`** achieves the best FID (41.26), suggesting that **residual connections improve generative quality** by preserving more global structure.  
- **`vae_std_z40`** shows minimal improvement over `vae_std_z20`, indicating that increasing latent space does not always enhance performance on simple datasets like MNIST.  
- **Batch Normalization** was tested but did not significantly improve results; however, it slowed training.

### **2. Convolutional VAE**
- **`vae_conv`** performs well in FID (50.13) but has higher MSE (0.0236) due to its limited training epochs.  
- Training is slower due to convolutional layers, and 10 epochs may not be enough for full convergence.

### **3. Effect of \(\beta\) in \(\beta\)-VAEs**
- Higher \(\beta\) values (e.g., \(\beta=4\), \(\beta=2\)) increase KL weight, penalizing reconstruction loss and leading to worse MSE but potentially better disentanglement.  
- Lower \(\beta\) (\(\beta=0.5\)) leads to better reconstruction (MSE = 0.0116) but does not significantly improve generation (high FID).  

### **4. WAE & GM-VAE**
- Both excel in reconstruction (lowest MSE: 0.0090-0.0092) but fail in generation (high FID).  
- WAE behaves more like an AE due to its architecture.  
- GM-VAE likely suffers from overfitting, leading to poor generalization despite strong reconstruction.

### **5. Additional Experiment: VQ-VAE**
- Training VQ-VAE was unstable and required extra tuning.  
- EMA optimization improved stability but required more epochs to converge (more than 10 epochs).  
- Due to limited compute resources, the full results are not included in this analysis, but code is available in [`models/vae.py`](models/vae.py) and [`models/vq.py`](models/vq.py).

---

## Conclusion


- For small datasets like MNIST, simpler models may generalize better and avoid overfitting.  
- If reconstruction is the primary goal, an AE (like WAE) is preferable to a VAE.  
- If generation quality matters, VAEs with residuals perform best.

This study highlights the inherent tradeoff between reconstruction fidelity and generative diversity. The choice of model should depend on whether the primary focus is high-quality generation (low FID) or accurate reconstruction (low MSE).

---

