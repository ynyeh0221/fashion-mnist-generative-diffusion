# (WIP) Fashion-MNIST Diffusion Model Implementation

This project implements a PyTorch-based diffusion model for generating Fashion-MNIST style clothing images. The model uses a U-Net architecture combined with time embeddings and class conditioning to enhance generation quality.

## Overview

- Train a diffusion model on the Fashion-MNIST dataset
- Enhance image generation quality with patch convolutions and self-attention
- Generate clothing images across 10 categories from random noise
- Visualize the complete generation process
- Support for model checkpoint saving and loading

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

System requirements:
- Python 3.6+
- PyTorch 1.7+
- CUDA-capable GPU (recommended but not required)

## Dataset

This project uses the Fashion-MNIST dataset, containing 28x28 grayscale images across 10 categories:
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

## Usage

### Training the Model

Run the script directly to start training:

```bash
python script_unet.py
```

### Configuration Options

You can modify the following parameters in the script:
- `epochs`: Number of training epochs (default 150)
- `num_diffusion_steps`: Number of diffusion steps (default 100)
- `batch_size`: Batch size (default 128)
- `img_size`: Image size (default 28x28)
- `patch_size`: Patch convolution size (default 2)

## Model Architecture

The implementation features:
- U-Net based encoder-decoder architecture
- Patch convolutions for efficient processing
- Linear-attention mechanism for enhanced generation quality
- Time step and class conditional embeddings
- Residual connections and batch normalization

## Output Files

The training process generates the following outputs:
1. Model checkpoints (saved every 10 epochs): `fashion_mnist_diffusion_epoch_{epoch}.pt`
2. Best model: `fashion_mnist_diffusion_best_model.pt`
3. Training loss plot: `training_loss.png`
4. Generation process visualizations (each epoch): `epoch_visualizations/epoch_{epoch}_generation.png`
5. Sample visualizations (each epoch): `epoch_visualizations/epoch_{epoch}_samples.png`
6. Final generated samples: `final_fashion_mnist_samples.png`

## Visualization Explanation

### Generation Graph
This visualization shows how the model progressively creates images from random noise. Each row represents a class, and each column represents a timestep in the denoising process. This helps understand the diffusion model's mechanics and track training progress.

### Sample Graph
This visualization displays multiple generated samples for each class. Each row represents a class, and each column represents different sample variations of that class. This helps evaluate the quality and diversity of the model's outputs.

## Code Structure

Main components:
- `ResBlock`: Basic residual block
- `PatchConv` & `PatchExpand`: Convolution layers for processing image patches
- `DoubleConv`: Double convolution block
- `Down` & `Up`: Downsampling and upsampling modules
- `SelfAttention`: Self-attention mechanism
- `FashionMNISTUNetDenoiser`: Main model class
- Diffusion process related functions: `q_sample`, etc.
- Visualization function: `generate_visualizations`

## Troubleshooting

If you encounter:
- Memory errors: Try reducing batch size or image size
- Training instability: Adjust learning rate, add more dropout, or tune beta values
- Poor generation quality: Increase training epochs or adjust network architecture

## Extension Ideas

- Try different U-Net architectures
- Add more conditional controls (like style, color)
- Implement classifier-free guidance to improve generation quality
- Add evaluation metrics (like FID score)

This diffusion model implementation provides a good starting point that can be extended and improved for application to other image generation tasks.
