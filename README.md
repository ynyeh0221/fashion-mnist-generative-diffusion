# fashion-mnist-generative-diffusion
A PyTorch implementation of a (latent/image) diffusion model for generating Fashion-MNIST-like digit images. The repository demonstrates both unconditional and conditional image generation, clearly showing how diffusion models remove noise step-by-step to generate digits from random noise or from a numeric prompt.

## Example Outputs

Below examples are generated by [LatentDiffusionModel v3](https://github.com/ynyeh0221/fashion-mnist-generative-diffusion/tree/main/LatentDiffusionModel/v3) of this repo.

### Diffusion Animation Gallery

Animations showing the diffusion process from clean image to noise and back, for various Fashion MNIST classes:

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
