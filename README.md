# Fashion-MNIST Generative Diffusion

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ynyeh0221/fashion-mnist-generative-diffusion)

This repository collects several PyTorch experiments on diffusion models trained on the Fashion-MNIST dataset.  Implementations range from image-space diffusion with CNNs and Transformers to a latent diffusion approach that combines an autoencoder with a conditional UNet.  Both unconditional and class-conditional generation are demonstrated.

## Repository structure
- **ImageDiffusionModel** – early experiments including CNN, Transformer and UNet based denoisers.
- **LatentDiffusionModel** – latent diffusion models that operate in a learned latent space.  The `v3` directory contains the most feature complete version.

Each subfolder contains its own training script and README describing the model and usage in more detail.

## Requirements
Install PyTorch and a few common libraries:
```bash
pip install torch torchvision matplotlib numpy tqdm scikit-learn
```
Python 3.8 or later and a CUDA-capable GPU are recommended for training but are not strictly required.

## Quick start
To train the latest latent diffusion model and generate example outputs:
```bash
cd LatentDiffusionModel/v3
python script_model_train_test.py
```
The script will automatically train the autoencoder and diffusion model if
no checkpoints are found and then produce a set of sample visualisations.
See the README inside each subdirectory for additional options and
explanations.

## Dataset
The models use the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset of 28×28 grayscale clothing images.  The dataset will be downloaded automatically by the provided scripts if it is not already present on your system.

## Example Outputs

Below examples are generated by [LatentDiffusionModel v3](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/tree/main/LatentDiffusionModel/v3) in this repository.

### Diffusion Animation Gallery

Animations showing the diffusion process from clean image to noise and back for various Fashion-MNIST classes:

#### Epoch 100

| T-shirt/Top | Trouser | Pullover | Dress | Coat |
|:-----------:|:-------:|:--------:|:-----:|:----:|
| ![T-shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_T-shirt-top_epoch_100.gif) | ![Trouser Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Trouser_epoch_100.gif) | ![Pullover Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Pullover_epoch_100.gif) | ![Dress Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Dress_epoch_100.gif) | ![Coat Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Coat_epoch_100.gif) |
| Sandal | Shirt | Sneaker | Bag | Ankle Boot |
| ![Sandal Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sandal_epoch_100.gif) | ![Shirt Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Shirt_epoch_100.gif) | ![Sneaker Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Sneaker_epoch_100.gif) | ![Bag Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Bag_epoch_100.gif) | ![Ankle Boot Animation](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/blob/main/LatentDiffusionModel/v3/diffusion_animation_class_Ankle%20boot_epoch_100.gif) |

Each animation shows the bidirectional diffusion process: starting with a clean generated image at `t=0`, adding noise up to `t=1000`, and then reversing back to the clean image.

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
