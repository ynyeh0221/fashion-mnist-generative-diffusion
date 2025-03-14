import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

# Set image size for Fashion MNIST (28x28 grayscale images)
img_size = 28

# Data loading and preprocessing for Fashion MNIST
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

# Load Fashion MNIST dataset
batch_size = 128
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Define class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Define a basic ResBlock for potential use in refinement
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):  # Added dropout
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)  # Added dropout

        # Skip connection with projection if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Create a new class for handling patches in convolutional layers
class PatchConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, stride=1, padding=0):  # Changed patch_size default to 2
        super(PatchConv, self).__init__()
        self.patch_size = patch_size

        # Create a convolution that operates on patches
        self.conv = nn.Conv2d(
            in_channels * patch_size * patch_size,  # Input channels × patch area
            out_channels,
            kernel_size=1,  # 1x1 convolution on the flattened patches
            stride=1,
            padding=0
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Ensure dimensions are divisible by patch_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            pad_h = self.patch_size - (height % self.patch_size)
            pad_w = self.patch_size - (width % self.patch_size)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            batch_size, channels, height, width = x.shape

        # Reshape to extract patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, height // self.patch_size, width // self.patch_size,
                                self.patch_size * self.patch_size)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, channels * self.patch_size * self.patch_size,
                                                       height // self.patch_size, width // self.patch_size)

        # Apply convolution to the flattened patches
        x = self.conv(x)

        return x


# Add a class for unpatchifying - the reverse operation of patching
class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2):  # Changed patch_size default to 2
        super(PatchExpand, self).__init__()
        self.patch_size = patch_size

        # Create a convolution to map patch features to pixel features
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * patch_size * patch_size,  # Output channels × patch area
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Apply 1x1 convolution to expand features
        x = self.conv(x)

        # Reshape to expand spatial dimensions
        channels_per_pixel = x.shape[1] // (self.patch_size * self.patch_size)
        x = x.view(batch_size, channels_per_pixel, self.patch_size, self.patch_size, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, channels_per_pixel, height * self.patch_size, width * self.patch_size)

        return x


# Modify the DoubleConv class to use patch convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, patch_size=1,
                 dropout_rate=0.1):  # Added dropout
        super().__init__()
        self.residual = residual
        self.patch_size = patch_size

        if not mid_channels:
            mid_channels = out_channels

        # First convolution using patches if patch_size > 1
        if patch_size > 1:
            self.conv1 = PatchConv(in_channels, mid_channels, patch_size=patch_size)
        else:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.GroupNorm(1, mid_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Added dropout

        # Second convolution using normal 3x3 conv to maintain spatial information
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Added dropout

    def forward(self, x):
        # Apply the first patch-based convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)  # Apply dropout

        # Apply the second standard convolution
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)  # Apply dropout

        if self.residual and x.shape == out.shape:
            return F.gelu(x + out)
        else:
            return F.gelu(out)


# Modify the Down class to incorporate patch convolutions
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, patch_size=2):  # Changed patch_size default to 2
        super().__init__()

        self.patch_size = patch_size

        # Replace maxpool with patch-based downsampling
        if patch_size > 1:
            self.patch_down = PatchConv(in_channels, in_channels, patch_size=patch_size)
            self.double_conv1 = DoubleConv(in_channels, in_channels, residual=True, patch_size=1)
            self.double_conv2 = DoubleConv(in_channels, out_channels, patch_size=1)
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, in_channels, residual=True),
                DoubleConv(in_channels, out_channels),
            )

        # Time and class embedding projection
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t_emb):
        if self.patch_size > 1:
            x = self.patch_down(x)
            x = self.double_conv1(x)
            x = self.double_conv2(x)
        else:
            x = self.maxpool_conv(x)

        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Modify the Up class to use patch-based upsampling
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, patch_size=2):  # Changed patch_size default to 2
        super().__init__()

        self.patch_size = patch_size

        # Use patch expansion if patch_size > 1, otherwise use bilinear upsampling
        if patch_size > 1:
            self.up = PatchExpand(in_channels // 2, in_channels // 2, patch_size=patch_size)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Convolutions using patch size
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Time and class embedding projection
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)

        # Handle different input sizes (ensure feature maps have same dimensions)
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate skip connections
        x = torch.cat([skip_x, x], dim=1)

        # Apply convolutions
        x = self.conv(x)

        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Self-attention block (can be included in the U-Net architecture)
class SelfAttention(nn.Module):
    """
    Linear Attention module with O(n) complexity instead of O(n²)
    """

    def __init__(self, channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = (channels // num_heads) * 3
        self.inner_dim = self.head_dim * num_heads

        self.query = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.key = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.value = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1)

        self.norm_q = nn.GroupNorm(1, channels)
        self.norm_k = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)

        self.attention_maps = None

    def forward(self, x):
        """
        Forward pass using linear attention mechanism
        """
        batch_size, c, h, w = x.shape

        q = self.query(self.norm_q(x))
        k = self.key(self.norm_k(x))
        v = self.value(self.norm_v(x))

        q = q.view(batch_size, self.num_heads, self.head_dim, h * w)
        k = k.view(batch_size, self.num_heads, self.head_dim, h * w)
        v = v.view(batch_size, self.num_heads, self.head_dim, h * w)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        k_sum = k.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        k_normalized = k / k_sum

        # (B, H, D, N) x (B, H, D, N) -> (B, H, D, D)
        context = torch.matmul(v, k_normalized.transpose(-2, -1))

        # (B, H, D, D) x (B, H, D, N) -> (B, H, D, N)
        out = torch.matmul(context, q)

        out = out.view(batch_size, self.inner_dim, h, w)
        out = self.out_proj(out)

        return out


# U-Net Model with time and class conditioning
class FashionMNISTUNetDenoiser(nn.Module):
    def __init__(self, img_size=28, in_channels=1, num_classes=10, time_dim=256,
                 attention_heads=8, patch_size=2):  # Changed patch_size default to 2
        super(FashionMNISTUNetDenoiser, self).__init__()

        # Store important parameters
        self.img_size = img_size
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.patch_size = patch_size

        # Time embedding network: converts scalar timestep to time_dim vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Class embedding: learns a vector for each class label
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial convolutional layer (standard conv, not patch-based)
        self.init_conv = DoubleConv(in_channels, 64, patch_size=1)

        # Encoder (downsampling path) with patch convolutions
        self.down1 = Down(64, 128, emb_dim=time_dim, patch_size=patch_size)
        self.sa1 = SelfAttention(128, num_heads=attention_heads)
        self.down2 = Down(128, 256, emb_dim=time_dim, patch_size=patch_size)
        self.sa2 = SelfAttention(256, num_heads=attention_heads)
        self.down3 = Down(256, 256, emb_dim=time_dim, patch_size=patch_size)
        self.sa3 = SelfAttention(256, num_heads=attention_heads)

        # Bottleneck layers at the lowest resolution (use standard convs)
        self.bottleneck1 = DoubleConv(256, 512, patch_size=1)
        self.bottleneck2 = DoubleConv(512, 512, patch_size=1)
        self.bottleneck3 = DoubleConv(512, 256, patch_size=1)

        # Decoder (upsampling path) with patch convolutions
        self.up1 = Up(512, 128, emb_dim=time_dim, patch_size=patch_size)
        self.sa4 = SelfAttention(128, num_heads=attention_heads)
        self.up2 = Up(256, 64, emb_dim=time_dim, patch_size=patch_size)
        self.sa5 = SelfAttention(64, num_heads=attention_heads)
        self.up3 = Up(128, 64, emb_dim=time_dim, patch_size=patch_size)
        self.sa6 = SelfAttention(64, num_heads=attention_heads)

        # Final output convolution (standard conv)
        self.final_conv = nn.Sequential(
            DoubleConv(64, 64, patch_size=1),
            nn.Conv2d(64, in_channels, kernel_size=1)  # 1x1 conv to map to grayscale
        )

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, noise_level, labels):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
            noise_level (torch.Tensor): Noise level tensor (timestep in diffusion process)
            labels (torch.Tensor): Class labels for conditioning

        Returns:
            torch.Tensor: Predicted noise with same shape as input
        """
        # Store original dimensions for ensuring output size matches input
        orig_height, orig_width = x.shape[2], x.shape[3]

        # Process noise level - handle different formats
        batch_size = x.shape[0]

        # Handle different noise_level formats
        if noise_level.dim() == 0 or (noise_level.dim() == 1 and noise_level.size(0) == 1):
            # It's a scalar or single-element tensor, expand to batch size
            noise_level = noise_level.view(1).expand(batch_size)
        elif noise_level.dim() == 1 and noise_level.size(0) == batch_size:
            # Already batch-sized but needs reshaping
            pass
        elif noise_level.dim() >= 2:
            # Flatten multi-dimensional tensor
            noise_level = noise_level.view(-1)
            if len(noise_level) == 1:
                noise_level = noise_level.expand(batch_size)
            elif len(noise_level) == batch_size:
                pass
            else:
                raise ValueError(f"Noise level shape {noise_level.shape} cannot be matched to batch size {batch_size}")

        # Ensure final shape is [batch_size, 1]
        noise_level = noise_level.view(batch_size, 1)
        t_emb = self.time_mlp(noise_level)

        # Process class label
        c_emb = self.class_embedding(labels)

        # Combine time and class embeddings
        combined_emb = t_emb + c_emb

        # Initial convolution
        x1 = self.init_conv(x)  # [B, 64, H, W]

        # Encoder path with self-attention
        x2 = self.down1(x1, combined_emb)  # [B, 128, H/2, W/2]
        x2 = self.sa1(x2)
        x3 = self.down2(x2, combined_emb)  # [B, 256, H/4, W/4]
        x3 = self.sa2(x3)
        x4 = self.down3(x3, combined_emb)  # [B, 256, H/8, W/8]
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bottleneck1(x4)  # [B, 512, H/8, W/8]
        x4 = self.bottleneck2(x4)  # [B, 512, H/8, W/8]
        x4 = self.bottleneck3(x4)  # [B, 256, H/8, W/8]

        # Decoder path with skip connections and self-attention
        x = self.up1(x4, x3, combined_emb)  # [B, 128, H/4, W/4]
        x = self.sa4(x)
        x = self.up2(x, x2, combined_emb)  # [B, 64, H/2, W/2]
        x = self.sa5(x)
        x = self.up3(x, x1, combined_emb)  # [B, 64, H, W]
        x = self.sa6(x)

        # Final convolution
        output = self.final_conv(x)  # [B, in_channels, H, W]

        # Ensure output size matches input size using interpolation if needed
        if output.shape[2] != orig_height or output.shape[3] != orig_width:
            output = F.interpolate(output, size=(orig_height, orig_width), mode='bilinear', align_corners=True)

        # Return predicted noise directly without the refinement step
        return output  # This now returns noise prediction, not denoised image

    def get_attention_maps(self):
        """
        Collect attention maps from all self-attention layers for visualization.

        Returns:
            list: List of attention maps from each self-attention layer
        """
        attention_maps = []
        for module in [self.sa1, self.sa2, self.sa3, self.sa4, self.sa5, self.sa6]:
            if hasattr(module, 'attention_maps'):
                attention_maps.append(module.attention_maps)
        return attention_maps


# Enhanced training with improved schedules and more diffusion steps
if __name__ == '__main__':
    # Initialize model, optimizer, loss function, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Current using device: {device}")

    # Create a U-Net model for Fashion MNIST
    model = FashionMNISTUNetDenoiser(
        img_size=img_size,
        in_channels=1,  # Grayscale
        num_classes=10,  # Fashion MNIST has 10 classes
        time_dim=256,
        patch_size=2  # Changed from 4 to 2
    ).to(device)

    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02,
                            betas=(0.9, 0.99))  # Increased from 8e-5 to 2e-4

    # Simplified loss function
    mse_loss = nn.MSELoss()


    # Learning rate scheduler with warmup
    def warmup_cosine_schedule(step, warmup_steps, max_steps):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))


    # Improved diffusion process
    epochs = 150
    num_diffusion_steps = 100
    max_steps = epochs * len(train_loader)
    warmup_steps = max_steps // 10  # 10% warmup

    # Track losses for plotting
    training_losses = []

    # Create output directory for visualizations
    import os

    os.makedirs('epoch_visualizations', exist_ok=True)

    # Set up noise schedule (beta values for diffusion process)
    betas = torch.linspace(1e-4, 0.02, num_diffusion_steps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # For reverse process (generation)
    posterior_variance = betas * (1. - alphas_cumprod.roll(1)) / (1. - alphas_cumprod)
    posterior_variance[0] = betas[0]
    posterior_log_variance = torch.log(posterior_variance)
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod.roll(1)) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod.roll(1)) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    posterior_mean_coef1[0] = 0
    posterior_mean_coef2[0] = 1 / torch.sqrt(alphas[0])


    # Standard diffusion process noise addition function
    def q_sample(x_start, t, noise=None):
        """
        Forward diffusion process: add noise to the sample according to schedule
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Convert t to indices for the alphas
        t_indices = t.to(torch.long)

        # Get the corresponding alpha values
        a = sqrt_alphas_cumprod.index_select(0, t_indices)
        a = a.view(-1, 1, 1, 1)

        # Get the corresponding sigma values
        sigma = sqrt_one_minus_alphas_cumprod.index_select(0, t_indices)
        sigma = sigma.view(-1, 1, 1, 1)

        # Apply the diffusion formula
        return a * x_start + sigma * noise, noise


    # Function to generate and save visualizations
    def generate_visualizations(epoch_num):
        model.eval()
        with torch.no_grad():
            display_classes = list(range(10))
            fig, axes = plt.subplots(len(display_classes), num_diffusion_steps // 5 + 1,
                                     figsize=(20, 5 * len(display_classes)))

            for i, class_idx in enumerate(display_classes):
                torch.manual_seed(42 + class_idx)
                x_T = torch.randn((1, 1, img_size, img_size)).to(device)
                label = torch.tensor([class_idx], device=device)

                noise_img = x_T.cpu().permute(0, 2, 3, 1).squeeze()
                noise_img = (noise_img * 0.5 + 0.5).clip(0, 1)

                axes[i, 0].imshow(noise_img, cmap='gray')
                axes[i, 0].set_title(f"Start (Noise)")
                axes[i, 0].axis('off')

                x = x_T

                steps_to_show = list(range(num_diffusion_steps, 0, -5))

                for idx, t in enumerate(steps_to_show):
                    if idx >= axes.shape[1] - 1:
                        break

                    t_tensor = torch.tensor([t - 1], device=device)

                    noise_pred = model(x, t_tensor / num_diffusion_steps, label)

                    # print(f"Step {t} - noise_pred min: {noise_pred.min().item():.4f}, max: {noise_pred.max().item():.4f}, mean: {noise_pred.mean().item():.4f}")

                    alpha_t = alphas_cumprod[t - 1]
                    alpha_t_sqrt = torch.sqrt(alpha_t)
                    sigma_t = torch.sqrt(1 - alpha_t)

                    x_0_pred = (x - sigma_t * noise_pred) / alpha_t_sqrt
                    x_0_pred = torch.clamp(x_0_pred, -1, 1)

                    if t > 1:
                        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                        beta_t = betas[t - 1]

                        mean = (
                                torch.sqrt(alphas[t - 1]) * (1 - alphas_cumprod[t - 2]) / (
                                1 - alphas_cumprod[t - 1]) * x +
                                torch.sqrt(alphas_cumprod[t - 2]) * beta_t / (1 - alphas_cumprod[t - 1]) * x_0_pred
                        )

                        var = beta_t * (1 - alphas_cumprod[t - 2]) / (1 - alphas_cumprod[t - 1])
                        sigma = torch.sqrt(var)

                        x = mean + sigma * z
                    else:
                        x = x_0_pred

                    img_to_show = x.cpu().permute(0, 2, 3, 1).squeeze()
                    img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)

                    # 绘制去噪图像
                    axes[i, idx + 1].imshow(img_to_show, cmap='gray')
                    axes[i, idx + 1].set_title(f"Step {t}")
                    axes[i, 0].set_ylabel(f"{class_names[class_idx]}", size='large',
                                          rotation=0, labelpad=40, va='center', ha='right')
                    axes[i, idx + 1].axis('off')

            plt.suptitle(f"Fashion MNIST Generation Process - Epoch {epoch_num + 1}", fontsize=16)
            plt.subplots_adjust(left=0.1, wspace=0.1, hspace=0.2)
            plt.tight_layout()
            plt.savefig(f'epoch_visualizations/epoch_{epoch_num + 1}_generation.png', dpi=200)
            plt.close()

            num_samples = 5
            fig, axes = plt.subplots(10, num_samples, figsize=(15, 20))

            for i, class_idx in enumerate(range(10)):
                for sample_idx in range(num_samples):
                    torch.manual_seed(sample_idx * 10 + class_idx)
                    x = torch.randn((1, 1, img_size, img_size)).to(device)
                    label = torch.tensor([class_idx], device=device)

                    for t in range(num_diffusion_steps, 0, -1):
                        t_tensor = torch.tensor([t - 1], device=device)

                        noise_pred = model(x, t_tensor / num_diffusion_steps, label)

                        alpha_t = alphas_cumprod[t - 1]
                        alpha_t_sqrt = torch.sqrt(alpha_t)
                        sigma_t = torch.sqrt(1 - alpha_t)

                        x_0_pred = (x - sigma_t * noise_pred) / alpha_t_sqrt
                        x_0_pred = torch.clamp(x_0_pred, -1, 1)

                        if t > 1:
                            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                            beta_t = betas[t - 1]

                            mean = (
                                    torch.sqrt(alphas[t - 1]) * (1 - alphas_cumprod[t - 2]) / (
                                    1 - alphas_cumprod[t - 1]) * x +
                                    torch.sqrt(alphas_cumprod[t - 2]) * beta_t / (1 - alphas_cumprod[t - 1]) * x_0_pred
                            )

                            var = beta_t * (1 - alphas_cumprod[t - 2]) / (1 - alphas_cumprod[t - 1])
                            sigma = torch.sqrt(var)

                            x = mean + sigma * z
                        else:
                            x = x_0_pred

                    img_to_show = x.cpu().permute(0, 2, 3, 1).squeeze()
                    img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)

                    axes[i, sample_idx].imshow(img_to_show, cmap='gray')
                    axes[i, sample_idx].axis('off')

                    if sample_idx == 0:
                        axes[i, sample_idx].set_ylabel(f"{class_names[i]}",
                                                       size='large', rotation=0,
                                                       labelpad=40, va='center', ha='right')

            plt.suptitle(f"Generated Fashion MNIST Samples - Epoch {epoch_num + 1}", fontsize=16)
            plt.subplots_adjust(left=0.1, wspace=0.1, hspace=0.1)
            plt.tight_layout()
            plt.savefig(f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png', dpi=200)
            plt.close()

        model.train() # Back to train mode


    # Main training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        torch.backends.cudnn.benchmark = True  # Speed up training

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            max_t = min(int((epoch / epochs) * num_diffusion_steps) + 10, num_diffusion_steps)
            t = torch.randint(1, max_t + 1, (images.shape[0],), device=device)

            # Add noise according to the standard diffusion process
            noisy_images, target_noise = q_sample(images, t - 1)  # t-1 for 0-indexed tensor

            # Normalize timesteps to [0,1]
            noise_level = ((t - 1) / num_diffusion_steps).view(-1, 1)

            optimizer.zero_grad()
            # Model now predicts the noise instead of the denoised image
            predicted_noise = model(noisy_images, noise_level, labels)

            # Simple MSE loss between predicted and target noise
            loss = mse_loss(predicted_noise, target_noise)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update learning rate with warmup cosine schedule
            global_step = epoch * len(train_loader) + step
            lr_scale = warmup_cosine_schedule(global_step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * 2e-4  # Increased base learning rate

            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if step % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Generate and save visualizations after each epoch
        if (epoch + 1) % 1 == 0:  # Generate visualizations every epoch
            generate_visualizations(epoch)

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'fashion_mnist_diffusion_epoch_{epoch + 1}.pt')

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

    # Generate final visualizations
    generate_visualizations(epochs - 1)

    print("Training completed!")

    # Create a visualization comparing epochs
    # (Moved outside any function, proper indentation level)
    with torch.no_grad():
        # Generate a large grid of final samples across all Fashion MNIST classes
        num_samples = 8
        fig, axes = plt.subplots(10, num_samples, figsize=(20, 25))  # All 10 classes

        for i, class_idx in enumerate(range(10)):  # All 10 classes
            for sample_idx in range(num_samples):
                # Use different seeds for variety
                torch.manual_seed(sample_idx * 10 + class_idx * 100)
                x = torch.randn((1, 1, img_size, img_size)).to(device)  # Grayscale
                label = torch.tensor([class_idx], device=device)

                # Perform full denoising process
                for t in range(num_diffusion_steps, 0, -1):
                    t_tensor = torch.tensor([t - 1], device=device)

                    # Predict noise
                    noise_pred = model(x, t_tensor / num_diffusion_steps, label)

                    # Calculate denoised x_0 prediction
                    alpha_t = alphas_cumprod[t - 1]
                    alpha_t_sqrt = torch.sqrt(alpha_t)
                    sigma_t = torch.sqrt(1 - alpha_t)

                    if t > 1:
                        # For t>1, we sample from the posterior
                        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                        beta_t = betas[t - 1]

                        # Get posterior mean
                        x_0_pred = (x - sigma_t * noise_pred) / alpha_t_sqrt
                        x_0_pred = torch.clamp(x_0_pred, -1, 1)

                        # Formula for posterior mean
                        mean = (
                                torch.sqrt(alphas[t - 1]) * (1 - alphas_cumprod[t - 2]) / (
                                    1 - alphas_cumprod[t - 1]) * x +
                                torch.sqrt(alphas_cumprod[t - 2]) * beta_t / (1 - alphas_cumprod[t - 1]) * x_0_pred
                        )

                        # Formula for posterior variance
                        var = beta_t * (1 - alphas_cumprod[t - 2]) / (1 - alphas_cumprod[t - 1])
                        sigma = torch.sqrt(var)

                        # Sample from posterior
                        x = mean + sigma * z
                    else:
                        # For the final step, we just use the predicted x_0
                        x = (x - sigma_t * noise_pred) / alpha_t_sqrt
                        x = torch.clamp(x, -1, 1)

                # Show final image
                img_to_show = x.cpu().permute(0, 2, 3, 1).squeeze()
                img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)  # Convert from [-1,1] to [0,1]

                axes[i, sample_idx].imshow(img_to_show, cmap='gray')  # Use grayscale colormap
                axes[i, sample_idx].axis('off')

                if sample_idx == 0:
                    axes[i, sample_idx].set_ylabel(f"{class_names[i]}",
                                                   size='large', rotation=0,
                                                   labelpad=40, va='center', ha='right')

        plt.suptitle("Final Generated Fashion MNIST Samples", fontsize=16)
        plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
        plt.tight_layout()
        plt.savefig('final_fashion_mnist_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
