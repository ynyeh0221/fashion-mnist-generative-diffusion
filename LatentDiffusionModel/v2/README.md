# Fashion MNIST Latent Diffusion Model

A PyTorch implementation of a Latent Diffusion Model (LDM) for generating Fashion MNIST images.

## Overview

This project implements a state-of-the-art Latent Diffusion Model for the Fashion MNIST dataset. The model combines a Variational Autoencoder (VAE) with a diffusion model in the latent space to generate high-quality fashion images. The implementation features various optimizations for training stability and generation quality.

## Architecture

The architecture consists of three main components:

### 1. Variational Autoencoder (VAE)

The VAE compresses the input images into a lower-dimensional latent space and reconstructs them back. It serves as the foundation for our latent diffusion process.

**Why VAE?**
- Provides an efficient compression of image data into a lower-dimensional latent space
- Learns a meaningful and continuous latent representation that captures the data distribution
- Enables working with a more compact representation, making the diffusion process more computationally efficient
- The probabilistic nature helps create smoother latent spaces compared to deterministic autoencoders

### 2. Latent UNet Denoiser

A U-Net architecture that operates in the latent space to denoise the latent representations.

**Why U-Net?**
- Skip connections help preserve spatial information at different resolutions
- Effectively captures both local and global context through its multi-scale architecture
- Proven effectiveness in image-to-image translation tasks
- The encoder-decoder structure is well-suited for the denoising process

**Enhanced with:**
- Self-attention mechanisms for improved feature learning
- Class conditioning to enable controlled generation
- Time embedding for noise level awareness
- Residual connections for better gradient flow

### 3. Latent Diffusion Model

Combines the VAE and the denoiser to create a complete generative model.

**Why Latent Diffusion?**
- Operates in the compressed latent space rather than pixel space, significantly reducing computational requirements
- Preserves the generative power of diffusion models while being more efficient
- Separates perceptual compression (VAE) from the generative process (diffusion)
- Enables higher quality generation with fewer diffusion steps

## Technical Features

### Optimized Training Process

1. **Two-Stage Training**
   - VAE is trained first with reconstruction and KL divergence losses
   - Diffusion model is trained second with frozen VAE parameters

2. **Cosine Noise Schedule**
   - Replaces the traditional linear noise schedule with a cosine schedule
   - Provides better sampling quality and training stability

3. **Hybrid Loss Function**
   - Combines MSE loss with cosine similarity loss
   - Weighted by noise level to focus more on low-noise steps
   - Ensures stable gradients and better convergence

4. **Exponential Moving Average (EMA)**
   - Maintains a moving average of model weights
   - Results in more stable and higher quality generation

5. **Adaptive Learning Rate**
   - Warmup period followed by cosine decay
   - Prevents unstable updates early in training

### Improved Sampling with DDIM

The model uses Denoising Diffusion Implicit Models (DDIM) sampling instead of the original DDPM:

- Faster sampling with fewer steps (50 vs 1000)
- Deterministic sampling option (η=0)
- Better quality generation with the same number of steps
- More stable denoising process

## Data Preparation

The model works with the Fashion MNIST dataset, which consists of 28×28 grayscale images of clothing items across 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset is automatically downloaded and preprocessed using torchvision.

## Usage

### Installation

```bash
pip install torch torchvision matplotlib numpy
```

### Training

```bash
python fashion_mnist_ldm.py
```

The training process happens in two phases:
1. VAE training (50 epochs)
2. Latent diffusion model training (200 epochs)

### Generating Images

After training, you can generate images for each class using the following:

```python
# Load the trained model
model_data = torch.load('fashion_mnist_latent_diffusion_final.pt')

# Create model instances and load weights
vae = VAE(in_channels=1, latent_dim=32).to(device)
vae.load_state_dict(model_data['vae'])

denoiser = LatentUNetDenoiser(latent_dim=32, num_classes=10, time_dim=256).to(device)
denoiser.load_state_dict(model_data['ema_denoiser'])  # Use EMA weights for better quality

latent_model = LatentDiffusionModel(vae, denoiser, latent_dim=32).to(device)

# Generate samples
generate_latent_diffusion_samples(latent_model, 0, 50, alphas_cumprod, betas)
```

## Results

The model generates high-quality Fashion MNIST images with the following characteristics:
- Clear category-specific features
- Sharp details and textures
- Diverse variations within each category
- Stable generation across all classes

Visualizations are saved during training showing:
- VAE reconstructions
- Generated samples for each fashion category
- Progressive improvement over training epochs

## Technical Considerations

### Numerical Stability

The implementation includes several measures to ensure numerical stability:
- Gradient clipping
- Value clamping in latent space
- NaN detection and handling
- Careful initialization of model weights
- Adaptive batch normalization

### Performance Optimizations

- Increased batch size for more stable gradients
- Simplified model architecture for faster training
- Optimized memory usage through spatial latent representation
- Balanced hyperparameters for quality vs. speed tradeoffs
