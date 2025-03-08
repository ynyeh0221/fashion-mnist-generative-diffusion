# Efficient Transformer-Based Diffusion Model for Fashion MNIST

This repository contains an implementation of a lightweight, transformer-based diffusion model for generating Fashion MNIST images. The model uses an efficient transformer architecture with optimizations for faster training and inference while maintaining good generation quality.

## Features

- **Patch-Based Approach**: Similar to Vision Transformers (ViT), uses image patches instead of individual pixels
- **Efficient Attention Mechanism**: Implements Linear Attention for better computational efficiency
- **Conditional Generation**: Generates images conditioned on class labels
- **Multi-Step Diffusion Process**: Employs a progressive denoising approach
- **Optimized Architecture**: Reduced model complexity with fewer layers and optimized hyperparameters

## Requirements

- PyTorch
- torchvision
- matplotlib
- GPU (optional, but recommended for faster training)

## Installation

```bash
pip install torch torchvision matplotlib
```

## Usage

Simply run the script to train the model and generate Fashion MNIST images:

```bash
python fashion_mnist_diffusion.py
```

The script will:
1. Download the Fashion MNIST dataset
2. Train the diffusion model for 30 epochs
3. Generate images for all 10 Fashion MNIST classes
4. Save the generation process visualization as 'fashion_mnist_generation.png'

## Model Architecture

The model consists of several components:

- **PatchEmbedding**: Converts input images into patch embeddings
- **PositionalEncoding**: Adds positional information to the embeddings
- **LinearAttention**: Efficient alternative to standard multi-head attention
- **EfficientTransformerBlock**: Transformer block with reduced complexity
- **EfficientTransformerDenoiser**: Main model that processes noisy images and predicts clean images

## Hyperparameters

- Image size: 16x16 pixels
- Patch size: 2x2 pixels
- Embedding dimension: 128
- Number of attention heads: 4
- Number of transformer layers: 3
- Feedforward dimension: 256
- Batch size: 128
- Learning rate: 3e-4 with cosine annealing warm restarts
- Weight decay: 0.05
- Number of diffusion steps: 10

## Fashion MNIST Classes

The model generates images for the following 10 classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Output Visualization

The script generates a visualization showing the progressive denoising process for each Fashion MNIST class, starting from random noise and ending with a clean generated image. This visualization is saved as 'fashion_mnist_generation.png'.

## Performance Optimizations

1. Adjusted input resolution for better balance between speed and detail
2. Decreased model complexity with smaller transformer blocks
3. Implemented patch-based approach similar to Vision Transformers
4. Employed efficient attention mechanism (Linear Attention)
5. Optimized batch size for better training stability
6. Improved learning rate scheduler with cosine annealing warm restarts
