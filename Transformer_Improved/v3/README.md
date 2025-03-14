# Transformer-Based Diffusion Model for Fashion MNIST Generation

This repository contains an implementation of an efficient transformer-based diffusion model for generating Fashion MNIST images. The model uses a combination of Vision Transformer (ViT) architecture and diffusion probabilistic models to generate high-quality fashion item images.

## Overview

This model implements a conditional denoising diffusion process using an efficient transformer architecture. It progressively transforms random noise into structured fashion images, conditioned on class labels. The architecture is optimized for performance while maintaining generation quality.

## Key Features

- **Patch-based Approach**: Images are processed as patches similar to Vision Transformers
- **Efficient Attention**: Uses linear attention mechanism for better computational efficiency
- **Conditional Generation**: Can generate specific fashion items based on class labels
- **Multi-step Diffusion Process**: Implements a step-by-step denoising process
- **Dynamic Loss Weighting**: Combines MSE and L1 losses with dynamic weighting

## Model Architecture

The model consists of several key components:

1. **PatchEmbedding**: Converts images into patch embeddings
2. **PositionalEncoding**: Adds positional information to the embeddings
3. **LinearAttention**: An efficient alternative to standard attention mechanism
4. **EfficientTransformerBlock**: Optimized transformer blocks with residual connections
5. **EfficientTransformerDenoiser**: The main model that combines all components

## Requirements

```
torch>=1.10.0
torchvision>=0.11.0
matplotlib>=3.4.0
```

## Usage

### Training

```python
# Initialize model
model = EfficientTransformerDenoiser(img_size=16)

# Configure optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1, eta_min=1e-6
)

# Train model
# See main script for full training loop implementation
```

### Generation

```python
# Generate a fashion item
model.eval()
with torch.no_grad():
    # Start with random noise
    image = torch.randn((1, 1, img_size, img_size))
    label = torch.tensor([class_idx])  # 0-9 for different fashion items
    
    # Perform denoising steps
    for t in reversed(range(1, num_diffusion_steps + 1)):
        noise_level = torch.tensor([[[[t / num_diffusion_steps]]]])
        image = model(image, noise_level, label)
        
    # Image now contains the generated fashion item
```

## Performance Optimizations

The model includes several optimizations for better efficiency:

- **Linear Attention**: O(n) complexity vs O(n²) in standard attention
- **Patch-based Processing**: Reduces sequence length compared to pixel-level
- **Reduced Model Complexity**: Fewer layers and smaller embedding dimensions
- **Gradient Clipping**: Enhances training stability
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

## Visualization

The model includes functionality to visualize the generation process, showing the progressive denoising from random noise to a fashion item. The visualization shows all 10 Fashion MNIST classes:

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

## Output Example

The model generates a visualization showing the step-by-step denoising process for each fashion category, saved as `fashion_mnist_generation.png`.
