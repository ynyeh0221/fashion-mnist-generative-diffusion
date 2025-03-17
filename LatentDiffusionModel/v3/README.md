# Fashion MNIST Conditional Diffusion Model

This project implements a class-conditional diffusion model for generating Fashion MNIST images. It combines an autoencoder to learn a compressed latent space representation with a diffusion model that can generate new images conditioned on specific fashion item classes.

## Overview

The architecture consists of two main components:

1. **Autoencoder** - Compresses the Fashion MNIST images into a latent space, enabling more efficient diffusion modeling.
2. **Conditional Diffusion Model** - Generates new fashion items by progressively denoising random noise, guided by class conditioning.

## Features

- **Class-conditional image generation** - Generate specific clothing items like t-shirts, dresses, or shoes
- **Attention mechanisms** - Enhanced feature selection using Channel Attention Layers (CAL)
- **Visualization tools** - Extensive visualization of the latent space, generation process, and class distributions
- **Progressive denoising** - Step-by-step visualization of how images form during the diffusion process

## Model Architecture

### Autoencoder
- **Encoder**: Convolutional layers with attention modules to compress images to a 64-dimensional latent space
- **Decoder**: Transposed convolutions to reconstruct images from the latent representation

### Diffusion Model
- **UNet backbone** with residual connections
- **Time embedding** using sinusoidal positional encoding
- **Class embedding** layer for conditioning on specific fashion categories
- **Self-attention mechanisms** in the bottleneck for global feature correlation

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn

## Usage

### Training

```python
# Main training function
python main.py
```

The training process includes:
1. First training the autoencoder to compress Fashion MNIST images
2. Then training the conditional diffusion model on the learned latent space

### Generating Images

```python
# Generate samples for a specific class
samples = generate_class_samples(
    autoencoder, 
    diffusion, 
    target_class="Dress",  # or class index (3)
    num_samples=5,
    save_path="generated_dresses.png"
)
```

### Visualizations

The model provides several visualization functions:

```python
# Visualize the entire generation process for a specific class
visualize_denoising_steps(
    autoencoder,
    diffusion,
    class_idx=5,  # 5 corresponds to "Sandal"
    save_path="sandal_generation_process.png"
)

# Generate a grid of samples for all classes
generate_samples_grid(
    autoencoder, 
    diffusion, 
    n_per_class=5,
    save_dir="./results"
)
```

## Example Outputs

### Diffusion Animation Gallery

Animations showing the diffusion process from clean image to noise and back, for various Fashion MNIST classes:

#### Epoch 5

| T-shirt/Top | Trouser | Pullover | Dress | Coat |
|:-----------:|:-------:|:--------:|:-----:|:----:|
| ![T-shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_T-shirt-top_epoch_5.gif) | ![Trouser Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Trouser_epoch_5.gif) | ![Pullover Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Pullover_epoch_5.gif) | ![Dress Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Dress_epoch_5.gif) | ![Coat Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Coat_epoch_5.gif) |
| Sandal | Shirt | Sneaker | Bag | Ankle Boot |
| ![Sandal Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sandal_epoch_5.gif) | ![Shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Shirt_epoch_5.gif) | ![Sneaker Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sneaker_epoch_5.gif) | ![Bag Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Bag_epoch_5.gif) | ![Ankle Boot Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Ankle%20boot_epoch_5.gif) |

#### Epoch 100

| T-shirt/Top | Trouser | Pullover | Dress | Coat |
|:-----------:|:-------:|:--------:|:-----:|:----:|
| ![T-shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_T-shirt-top_epoch_100.gif) | ![Trouser Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Trouser_epoch_100.gif) | ![Pullover Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Pullover_epoch_100.gif) | ![Dress Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Dress_epoch_100.gif) | ![Coat Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Coat_epoch_100.gif) |
| Sandal | Shirt | Sneaker | Bag | Ankle Boot |
| ![Sandal Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sandal_epoch_100.gif) | ![Shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Shirt_epoch_100.gif) | ![Sneaker Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sneaker_epoch_100.gif) | ![Bag Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Bag_epoch_100.gif) | ![Ankle Boot Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Ankle%20boot_epoch_100.gif) |

Each animation shows the bidirectional diffusion process - starting with a clean generated image (t=0), progressing to random noise (t=1000), and then reversing back to the clean image in a continuous loop.

### Denoising Process Visualization

The visualization shows both:
- The step-by-step denoising of the image (top)
- The corresponding path through the latent space (bottom)

#### Epoch 100

| T-shirt/Top | Trouser | Pullover | Dress | Coat |
|:-----------:|:-------:|:--------:|:-----:|:----:|
| ![T-shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_T-shirt-top_epoch_100.png) | ![Trouser Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Trouser_epoch_100.png) | ![Pullover Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Pullover_epoch_100.png) | ![Dress Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Dress_epoch_100.png) | ![Coat Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Coat_epoch_100.png) |
| Sandal | Shirt | Sneaker | Bag | Ankle Boot |
| ![Sandal Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Sandal_epoch_100.png) | ![Shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Shirt_epoch_100.png) | ![Sneaker Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Sneaker_epoch_100.png) | ![Bag Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Bag_epoch_100.png) | ![Ankle Boot Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/output/denoise_with_diffusion_path/denoising_path_Ankle%20boot_epoch_100.png) |

## Model Components

### SimpleAutoencoder
- Compresses images into a latent representation and reconstructs them
- Uses Channel Attention Layers for improved feature selection

### ConditionalUNet
- Predicts noise at each diffusion step
- Incorporates class conditioning to guide the generation process
- Uses time embedding to handle different diffusion timesteps

### ConditionalDenoiseDiffusion
- Manages the forward and reverse diffusion processes
- Handles class conditioning during sample generation

## Results

The model successfully learns to generate diverse, high-quality fashion items for each class, demonstrating:
- Good class separation in the latent space
- Consistent generation of class-specific features
- Smooth transitions during the denoising process
