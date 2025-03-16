import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os
import numpy as np
import copy

# Set image size for Fashion MNIST (28x28 grayscale images)
img_size = 28

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # This already scales to [0,1] range
])

# Load Fashion MNIST dataset
batch_size = 128  # Increased for more stable gradients
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Define class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Define VAE model
class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32):  # Reduced latent dimension
        super(VAE, self).__init__()

        # Simpler encoder - fewer layers, no residual connections
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.LeakyReLU(0.2),
        )

        # Calculate the flattened dimension correctly
        self.flatten_dim = 128 * 7 * 7

        # Projection to latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 128 * 7 * 7)

        # Simpler decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()  # Enforce output range [0,1] for stability
        )

        self.latent_dim = latent_dim
        # Control parameter for KL weight
        self.kl_weight = 0.005

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 128, 7, 7)
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


# Define basic ResBlock for the latent space denoiser
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Use projection shortcut if dimensions change
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
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Double convolution module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dropout_rate=0.1):
        super().__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, mid_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        if self.residual and x.shape == out.shape:
            return F.gelu(x + out)
        else:
            return F.gelu(out)


# Downsampling module
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

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
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Upsampling module with corrected channel dimensions
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # First DoubleConv takes concatenated channels (in_channels + skip_channels)
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Time and class embedding projection
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)

        # Handle different input sizes
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate skip connections
        x = torch.cat([skip_x, x], dim=1)

        # Apply convolutions
        x = self.conv(x)

        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Enhanced Self-attention module
class EnhancedSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(1, channels)  # Use GroupNorm which works well with conv nets

        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # Use 1x1 convolution
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Apply normalization
        x_norm = self.norm(x)

        # Compute QKV with 1x1 convolution
        qkv = self.qkv(x_norm)

        # Reshape and permute for attention calculation
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, b, heads, h*w, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine values using attention weights
        out = (attn @ v)

        # Reshape back to spatial format
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)

        # Final projection
        out = self.proj(out)

        return out + x  # Add residual connection


# Latent space U-Net denoiser for diffusion model
class LatentUNetDenoiser(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10, time_dim=256, attention_heads=8):
        super(LatentUNetDenoiser, self).__init__()

        # Store important parameters
        self.latent_dim = latent_dim
        self.time_dim = time_dim

        # Improved time embedding network with sinusoidal embeddings
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding: learns a vector for each class label
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial convolutional layer
        self.init_conv = DoubleConv(latent_dim, 64)

        # Encoder (downsampling path)
        self.down1 = Down(64, 192, emb_dim=time_dim)
        self.sa1 = EnhancedSelfAttention(192, num_heads=attention_heads)
        self.down2 = Down(192, 384, emb_dim=time_dim)
        self.sa2 = EnhancedSelfAttention(384, num_heads=attention_heads)
        self.down3 = Down(384, 384, emb_dim=time_dim)
        self.sa3 = EnhancedSelfAttention(384, num_heads=attention_heads)

        # Bottleneck layers
        self.bottleneck1 = DoubleConv(384, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 384)

        # Decoder (upsampling path) - with correct channel specifications
        # Parameters: (input_channels, skip_channels, output_channels)
        self.up1 = Up(384, 384, 192, emb_dim=time_dim)
        self.sa4 = EnhancedSelfAttention(192, num_heads=attention_heads)
        self.up2 = Up(192, 192, 64, emb_dim=time_dim)
        self.sa5 = EnhancedSelfAttention(64, num_heads=attention_heads)
        self.up3 = Up(64, 64, 64, emb_dim=time_dim)
        self.sa6 = EnhancedSelfAttention(64, num_heads=attention_heads)

        # Final output convolution
        self.final_conv = nn.Sequential(
            DoubleConv(64, 64),
            nn.Conv2d(64, latent_dim, kernel_size=1)  # 1x1 conv to map back to latent dimension
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, z, noise_level, labels):
        """
        Forward pass of the U-Net model.

        Args:
            z (torch.Tensor): Input tensor in latent space
            noise_level (torch.Tensor): Noise level tensor (timestep in diffusion process)
            labels (torch.Tensor): Class labels for conditioning

        Returns:
            torch.Tensor: Predicted noise with same shape as input
        """
        batch_size = z.shape[0]

        # Simplified noise level processing
        if isinstance(noise_level, (int, float)):
            noise_level = torch.tensor([noise_level], device=z.device).expand(batch_size, 1)
        else:
            if not isinstance(noise_level, torch.Tensor):
                noise_level = torch.tensor(noise_level, device=z.device)

            if noise_level.dim() == 0:  # Scalar tensor
                noise_level = noise_level.view(1, 1).expand(batch_size, 1)
            elif noise_level.dim() == 1:  # 1D tensor
                if noise_level.size(0) == 1:
                    noise_level = noise_level.view(1, 1).expand(batch_size, 1)
                elif noise_level.size(0) == batch_size:
                    noise_level = noise_level.view(batch_size, 1)
                else:
                    raise ValueError(f"Noise level length {noise_level.size(0)} does not match batch size {batch_size}")
            elif noise_level.dim() == 2 and noise_level.shape == (batch_size, 1):
                # Already in the correct shape [batch_size, 1]
                pass
            else:
                raise ValueError(
                    f"Noise level should be a scalar, 1D tensor, or shape [batch_size, 1], got shape {noise_level.shape}")

        # Ensure noise level is on the correct device
        noise_level = noise_level.to(z.device)

        # Process time embedding
        t_emb = self.time_mlp(noise_level)

        # Process class label embedding
        c_emb = self.class_embedding(labels)

        # Combine time and class embeddings
        combined_emb = t_emb + c_emb

        # Initial convolution
        z1 = self.init_conv(z)  # [B, 64, H, W]

        # Encoder path
        z2 = self.down1(z1, combined_emb)  # [B, 192, H/2, W/2]
        z2 = self.sa1(z2)
        z3 = self.down2(z2, combined_emb)  # [B, 384, H/4, W/4]
        z3 = self.sa2(z3)
        z4 = self.down3(z3, combined_emb)  # [B, 384, H/8, W/8]
        z4 = self.sa3(z4)

        # Bottleneck
        z4 = self.bottleneck1(z4)  # [B, 512, H/8, W/8]
        z4 = self.bottleneck2(z4)  # [B, 512, H/8, W/8]
        z4 = self.bottleneck3(z4)  # [B, 384, H/8, W/8]

        # Decoder path
        z = self.up1(z4, z3, combined_emb)  # [B, 192, H/4, W/4]
        z = self.sa4(z)
        z = self.up2(z, z2, combined_emb)  # [B, 64, H/2, W/2]
        z = self.sa5(z)
        z = self.up3(z, z1, combined_emb)  # [B, 64, H, W]
        z = self.sa6(z)

        # Final convolution
        output = self.final_conv(z)  # [B, latent_dim, H, W]

        # Return predicted noise
        return output


# Improved time embedding with sinusoidal positional encoding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # In case of odd dimension
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))

        return embeddings


# Main Latent Diffusion Model class
class LatentDiffusionModel(nn.Module):
    def __init__(self, vae, denoiser, latent_dim=32):
        super(LatentDiffusionModel, self).__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.latent_dim = latent_dim

    def encode(self, x):
        with torch.no_grad():
            mu, log_var = self.vae.encode(x)
            latent_spatial_size = 16
            z = mu.view(mu.size(0), mu.size(1), 1, 1)
            z = F.interpolate(z, size=(latent_spatial_size, latent_spatial_size), mode='bilinear', align_corners=True)
        return z, latent_spatial_size

    # Improved latent sampling method
    def sample_from_latent(self, x):
        with torch.no_grad():
            mu, log_var = self.vae.encode(x)

            # Use mean directly for more stable representation
            z = mu

            # Spread to spatial dimensions
            latent_spatial_size = 16
            z = z.view(z.size(0), z.size(1), 1, 1)
            z = F.interpolate(z, size=(latent_spatial_size, latent_spatial_size), mode='bilinear', align_corners=True)

            # Ensure latent is in a reasonable range
            z = torch.clamp(z, -5.0, 5.0)
        return z

    def decode(self, z):
        # Decode latent representation
        with torch.no_grad():
            # Better conversion from spatial to vector
            if len(z.shape) == 4 and z.shape[2] > 1:  # If z is in spatial form
                z_flat = F.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                z_flat = z

            reconstruction = self.vae.decode(z_flat)
        return reconstruction

    def forward(self, x, noise_level, labels):
        # Diffusion prediction in latent space
        return self.denoiser(x, noise_level, labels)


# New function: Cosine noise schedule
def get_cosine_noise_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# Updated learning rate schedule
def warmup_cosine_schedule(step, warmup_steps, max_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Higher minimum LR


# Helper function for EMA model management
def update_ema(target_params, source_params, decay_rate):
    for target, source in zip(target_params, source_params):
        target.data.copy_(target.data * decay_rate + source.data * (1 - decay_rate))


# FIXED: Updated train_vae function with MSE loss to avoid negative loss values
def train_vae(vae, num_epochs=50, lr=0.0001):
    import numpy as np
    print("Starting VAE training with MSE loss...")

    # Use vanilla Adam with very low learning rate
    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.9, 0.999))

    # Simple step decay learning rate schedule
    def lr_lambda(epoch):
        if epoch < 10:
            return 1.0  # First 10 epochs at full learning rate
        elif epoch < 20:
            return 0.5  # Next 10 epochs at half learning rate
        else:
            return 0.1  # Remaining epochs at 1/10 learning rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Track losses
    train_losses = []
    recon_losses = []
    kld_losses = []

    def get_kl_weight(epoch, num_epochs):
        warmup_epochs = num_epochs // 2
        if epoch < warmup_epochs:
            return 0.005 + (0.008 - 0.005) * (epoch / warmup_epochs)
        else:
            return 0.008

    # Using MSE loss which is guaranteed to be positive
    def reconstruction_loss(recon_x, x):
        """
        Using squared Euclidean distance (sum of squared differences)
        instead of MSE (mean of squared differences) to avoid small loss values
        """
        # Calculate squared differences
        squared_diff = (recon_x - x) ** 2

        # Sum all elements instead of taking the mean
        # This avoids division by width × height × channels
        return torch.sum(squared_diff)

    # KL divergence with proper reduction
    def kl_divergence(mu, log_var, kl_weight):
        """
        Standard KL divergence formula with proper reduction
        """
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        return kl_weight * kld

    device = next(vae.parameters()).device

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0

        # Update KL weight for this epoch
        kl_weight = get_kl_weight(epoch, num_epochs)
        vae.kl_weight = kl_weight  # Store for reference

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, log_var = vae(data)

            # Calculate losses
            recon_loss = reconstruction_loss(recon_batch, data)
            kld_loss = kl_divergence(mu, log_var, kl_weight)

            # Total loss with very small KL weight
            loss = recon_loss + kld_loss

            # Debug: track min/max values for checking
            if batch_idx % 100 == 0:
                print(f"Data range: [{data.min().item():.4f}, {data.max().item():.4f}], "
                      f"Recon range: [{recon_batch.min().item():.4f}, {recon_batch.max().item():.4f}]")

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} = Recon(MSE): {recon_loss.item():.4f} + '
                      f'KL*weight: {kld_loss.item():.4f} (weight={kl_weight:.6f})')

        # Update learning rate
        scheduler.step()

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kld = total_kld / len(train_loader)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)

        print(f'====> Epoch: {epoch + 1} Avg Loss: {avg_loss:.4f} = '
              f'Recon: {avg_recon:.4f} + KL*weight: {kl_weight:.6f}*{avg_kld:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Visualize reconstructions
        if (epoch + 1) % 1 == 0:
            visualize_reconstructions(vae, epoch)
        if (epoch + 1) % 5 == 0:
            # Save checkpoint
            save_checkpoint(vae, optimizer, epoch, avg_loss, f'vae_checkpoint_epoch_{epoch + 1}.pt')

    return vae, train_losses, recon_losses, kld_losses


# Modified train_latent_diffusion function
def train_latent_diffusion(vae, denoiser, latent_model, num_epochs=200, num_diffusion_steps=30):
    print("Starting latent diffusion model training...")

    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    # Increased learning rate and modified optimizer parameters
    optimizer = optim.AdamW(denoiser.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)

    # Add EMA setup
    ema_decay = 0.995
    ema_denoiser = copy.deepcopy(denoiser)  # Create a deep copy for EMA

    # Define loss function
    def hybrid_loss(pred, target, noise_level):
        """
        Modified hybrid loss using squared Euclidean distance instead of MSE
        to prevent loss values from becoming too small due to division by dimensions.
        """
        # Reshape weights for proper broadcasting
        weights = 1.0 / (1.0 + noise_level * 10.0)  # More weight to low-noise steps
        weights = weights.view(-1, 1, 1, 1)  # Reshape to [batch_size, 1, 1, 1]

        # Squared Euclidean distance component (NOT divided by dimensions)
        squared_error = (pred - target) ** 2
        weighted_squared_error = weights * squared_error

        # Use sum instead of mean - this is the key change
        euc_loss = torch.sum(weighted_squared_error)

        # Process each sample in the batch individually for cosine similarity
        batch_size = pred.shape[0]
        cos_loss_sum = 0.0

        # Flatten and compute cosine similarity per sample
        for i in range(batch_size):
            p = pred[i].view(-1)  # Flatten to 1D
            t = target[i].view(-1)  # Flatten to 1D

            # Skip samples with all zeros to avoid NaN
            if torch.all(torch.abs(p) < 1e-6) or torch.all(torch.abs(t) < 1e-6):
                cos_sim = 1.0  # No loss for zero vectors
            else:
                # Compute cosine similarity (-1 to 1)
                cos_sim = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0))

            # Convert to loss (0 to 2) and weight by noise level
            sample_cos_loss = weights[i].item() * (1.0 - cos_sim)
            cos_loss_sum += sample_cos_loss

        cos_loss = cos_loss_sum

        # Scale cosine loss to match the magnitude of Euclidean loss
        # This is important since we're now using sum instead of mean
        # which changes the scale of the Euclidean loss
        euc_scale = max(1.0, euc_loss.item() / (batch_size * 100))
        scaled_cos_loss = cos_loss * euc_scale

        # Balance the two loss components
        lambda_euc = 1  # Weight for Euclidean loss
        lambda_cos = 0  # Weight for cosine loss

        # Combine losses
        total_loss = lambda_euc * euc_loss + lambda_cos * scaled_cos_loss

        return total_loss

    # Training parameters
    max_steps = num_epochs * len(train_loader)
    warmup_steps = max_steps // 10

    # Use cosine noise schedule instead of linear
    betas = get_cosine_noise_schedule(num_diffusion_steps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = alphas_cumprod.clamp(min=1e-5)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Forward diffusion process (adding noise)
    def q_sample(z_start, t, noise=None):
        """Add noise in latent space"""
        if noise is None:
            noise = torch.randn_like(z_start)

        t_indices = t.to(torch.long)
        a = sqrt_alphas_cumprod.index_select(0, t_indices).view(-1, 1, 1, 1)
        sigma = sqrt_one_minus_alphas_cumprod.index_select(0, t_indices).view(-1, 1, 1, 1)

        return a * z_start + sigma * noise, noise

    # Add EMA update function
    def update_ema(target_params, source_params, decay_rate):
        for target, source in zip(target_params, source_params):
            target.data.copy_(target.data * decay_rate + source.data * (1 - decay_rate))

    training_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        denoiser.train()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Modified latent sampling - use direct mu for more stability
            with torch.no_grad():
                mu, log_var = vae.encode(images)
                z = mu  # Use mean directly instead of sampling for stability
                latent_spatial_size = 16
                z = z.view(z.size(0), z.size(1), 1, 1)
                z = F.interpolate(z, size=(latent_spatial_size, latent_spatial_size), mode='bilinear',
                                  align_corners=True)
                z = torch.clamp(z, -5.0, 5.0)  # Tighter clamping for stability

            # Randomly select timestep - modified to sample more from end of process
            t_weights = torch.linspace(1.0, 0.1, num_diffusion_steps).to(device)
            t_weights = t_weights / t_weights.sum()
            t = torch.multinomial(t_weights, images.shape[0], replacement=True) + 1

            # Add noise
            noisy_z, target_noise = q_sample(z, t - 1)

            # Normalize timestep - using non-linear transformation for better time representation
            noise_level = (1 - torch.cos((t - 1) / num_diffusion_steps * math.pi / 2)).view(-1, 1)

            optimizer.zero_grad()

            # Predict noise
            predicted_noise = denoiser(noisy_z, noise_level, labels)

            # Apply weighted loss based on noise level
            loss = hybrid_loss(predicted_noise, target_noise, noise_level)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN loss at epoch {epoch}, step {step}")
                continue  # skip this batch

            loss.backward()

            # Increased gradient clipping threshold for higher learning rate
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)

            # Update learning rate - modified schedule
            global_step = epoch * len(train_loader) + step
            lr_scale = warmup_cosine_schedule(global_step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * 5e-4  # Higher base learning rate

            optimizer.step()

            # Update EMA model
            update_ema(ema_denoiser.parameters(), denoiser.parameters(), ema_decay)

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f'Latent Diffusion Training: Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        print(f'Latent Diffusion Training: Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save model - save both regular and EMA model
        if (epoch + 1) % 10 == 0:
            torch.save(denoiser.state_dict(), f'fashion_mnist_latent_diffusion_epoch_{epoch + 1}.pt')
            torch.save(ema_denoiser.state_dict(), f'fashion_mnist_latent_diffusion_ema_epoch_{epoch + 1}.pt')

        # Save generated visualizations - use EMA model for more stable results
        if (epoch + 1) % 1 == 0:
            # Temporarily swap to EMA model for generation
            orig_state_dict = copy.deepcopy(denoiser.state_dict())
            denoiser.load_state_dict(ema_denoiser.state_dict())
            generate_latent_diffusion_samples(latent_model, epoch, num_diffusion_steps, alphas_cumprod, betas)
            for class_idx in range(10):
                visualize_latent_space_denoising(
                    latent_model,
                    num_diffusion_steps,
                    alphas_cumprod,
                    betas,
                    class_idx=class_idx
                )
            # Swap back
            denoiser.load_state_dict(orig_state_dict)

    print("Latent diffusion model training completed!")
    return denoiser, ema_denoiser


# FIXED: Updated helper function for visualization
def visualize_reconstructions(vae, epoch):
    vae.eval()
    with torch.no_grad():
        sample = next(iter(train_loader))[0][:8].to(next(vae.parameters()).device)
        recon, _, _ = vae(sample)

        # Create comparison visualization
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))

        for i in range(8):
            # Original
            img = sample[i].cpu().numpy().squeeze()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

            # Reconstruction
            recon_img = recon[i].cpu().numpy().squeeze()
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(f'vae_recon_epoch_{epoch + 1}.png')
        plt.close()
    vae.train()


def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


# Updated visualization function
def visualize_vae_reconstruction(vae, epoch):
    vae.eval()
    with torch.no_grad():
        # Get a batch of data
        test_batch, test_labels = next(iter(train_loader))
        test_batch = test_batch.to(device)

        # Reconstruct through VAE
        recon_batch, _, _ = vae(test_batch)

        # Visualize original and reconstructed images
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))

        for i in range(8):
            # Original image - no need to denormalize since we're not normalizing anymore
            img = test_batch[i].cpu().permute(1, 2, 0).squeeze().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Original {class_names[test_labels[i]]}")
            axes[0, i].axis('off')

            # Reconstructed image
            recon = recon_batch[i].cpu().permute(1, 2, 0).squeeze().numpy()
            axes[1, i].imshow(recon, cmap='gray')
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(f'vae_reconstruction_epoch_{epoch + 1}.png', dpi=200)
        plt.close()
    vae.train()

# Add this new function to your code
def dpm_solver_2_sampling(latent_model, num_diffusion_steps, alphas_cumprod, class_labels=None, batch_size=1,
                          num_sampling_steps=10):
    """
    Implements DPM-Solver-2 sampling algorithm for faster generation

    Args:
        latent_model: The latent diffusion model
        num_diffusion_steps: Total number of diffusion steps in the model
        alphas_cumprod: The pre-computed alphas cumulative product
        class_labels: Optional class conditioning labels
        batch_size: Number of samples to generate at once
        num_sampling_steps: Number of solver steps (much fewer than DDIM)

    Returns:
        Decoded images from the generated latent samples
    """
    vae = latent_model.vae
    denoiser = latent_model.denoiser
    latent_dim = latent_model.latent_dim
    latent_spatial_size = 16
    device = next(denoiser.parameters()).device

    # Set evaluation mode
    vae.eval()
    denoiser.eval()

    # Prepare model to generate samples
    with torch.no_grad():
        # Create initial latent noise
        z = torch.randn((batch_size, latent_dim, latent_spatial_size, latent_spatial_size)).to(device)
        z = torch.clamp(z, -4.0, 4.0)

        # Setup timestamps for solver
        timesteps = torch.linspace(1, 0, num_sampling_steps + 1).to(device) * (num_diffusion_steps - 1)
        timesteps = (timesteps.long())[:-1]

        # For tracking intermediate results if needed
        intermediate_results = []

        # Process class labels if provided
        if class_labels is None:
            # If no labels provided, use a tensor of zeros
            class_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        else:
            if isinstance(class_labels, torch.Tensor):
                class_labels = class_labels.clone().detach().to(device)
            else:
                class_labels = torch.tensor(class_labels, device=device)
            if class_labels.shape[0] == 1 and batch_size > 1:
                # Expand single label to batch size if needed
                class_labels = class_labels.expand(batch_size)

        # DPM-Solver-2 algorithm
        for i in range(len(timesteps) - 1):
            # Current and next time steps
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            # Normalized time for model input
            t_normalized = t_cur / num_diffusion_steps
            if isinstance(t_normalized, torch.Tensor):
                t_normalized_tensor = t_normalized.clone().detach().to(device)
            else:
                t_normalized_tensor = torch.tensor(t_normalized, device=device)
            noise_level_scalar = (1 - torch.cos(t_normalized_tensor * math.pi / 2)).item()
            noise_level = torch.full((batch_size, 1), noise_level_scalar, device=device)

            # Get noise prediction from model
            noise_pred = denoiser(z, noise_level, class_labels)
            noise_pred = torch.clamp(noise_pred, -5.0, 5.0)

            # Get alpha values for current timestep
            alpha_cur = alphas_cumprod[t_cur].clamp(min=1e-5)
            alpha_next = alphas_cumprod[t_next].clamp(min=1e-5)

            # Current x0 prediction
            x0_pred = (z - torch.sqrt(1 - alpha_cur) * noise_pred) / torch.sqrt(alpha_cur)
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

            # First-order update (DPM-Solver-1)
            if i == 0:
                # For the first step, use simpler update
                z = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
                intermediate_results.append(z.clone())
            else:
                # For subsequent steps, use second-order solver
                h = t_next - t_cur
                h_last = t_cur - timesteps[i - 1]

                # Get intermediate values needed for the second-order step
                r = h_last / h

                # Calculate intermediate latent for obtaining the second noise prediction
                z_mid = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred

                # Get noise prediction at the intermediate latent
                t_next_normalized = t_next / num_diffusion_steps
                if isinstance(t_next_normalized, torch.Tensor):
                    t_next_tensor = t_next_normalized.clone().detach().to(device)
                else:
                    t_next_tensor = torch.tensor(t_next_normalized, device=device)
                noise_level_next_scalar = (1 - torch.cos(t_next_tensor * math.pi / 2)).item()
                noise_level_next = torch.full((batch_size, 1), noise_level_next_scalar, device=device)
                noise_pred_next = denoiser(z_mid, noise_level_next, class_labels)
                noise_pred_next = torch.clamp(noise_pred_next, -5.0, 5.0)

                # Second predicted x0
                x0_pred_next = (z_mid - torch.sqrt(1 - alpha_next) * noise_pred_next) / torch.sqrt(alpha_next)
                x0_pred_next = torch.clamp(x0_pred_next, -10.0, 10.0)

                # Apply second-order correction
                # This weighted combination of x0 predictions gives a more accurate estimate
                z = torch.sqrt(alpha_next) * ((1 + 1 / r) * x0_pred_next - (1 / r) * x0_pred) + \
                    torch.sqrt(1 - alpha_next) * noise_pred_next

                # Safety clamp
                z = torch.clamp(z, -20.0, 20.0)

                # Store intermediate result if needed
                intermediate_results.append(z.clone())

        # Final step (t=0) - convert spatial latent back to vector for decoding
        z_flat = F.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)

        # Decode final latent representation to images
        images = vae.decode(z_flat)
        images = images.cpu()

    return images, intermediate_results

# Add this function to visualize the denoising progression
def visualize_latent_space_denoising(latent_model, num_diffusion_steps, alphas_cumprod, betas, class_idx=0):
    """
    Visualize the denoising process directly in latent space without decoding to images
    """
    import torch
    import matplotlib.pyplot as plt
    import os

    vae = latent_model.vae
    denoiser = latent_model.denoiser
    latent_spatial_size = 16
    latent_dim = latent_model.latent_dim
    device = next(denoiser.parameters()).device

    vae.eval()
    denoiser.eval()

    # DPM-Solver-2 sampling parameters
    num_sampling_steps = 10  # We'll visualize 10 steps

    with torch.no_grad():
        # Create figure for visualization
        fig, axes = plt.subplots(2, num_sampling_steps + 1, figsize=(24, 8))

        # Sample a latent vector from standard normal distribution
        z = torch.randn((1, latent_dim, latent_spatial_size, latent_spatial_size)).to(device)
        z = torch.clamp(z, -4.0, 4.0)  # Initial clamping

        # Set the class label
        label = torch.tensor([class_idx], device=device)

        # Save the initial noisy latent
        initial_z = z.clone()

        # Select a middle channel to visualize
        channel_to_vis = latent_dim // 2  # Middle channel

        # Visualize initial latent
        latent_vis = initial_z[0, channel_to_vis].cpu().numpy()
        axes[0, 0].imshow(latent_vis, cmap='viridis')
        axes[0, 0].set_title(f"Initial Latent (channel {channel_to_vis})")
        axes[0, 0].axis('off')

        # Also show the standard deviation across channels to see overall structure
        latent_std = torch.std(initial_z[0], dim=0).cpu().numpy()
        axes[1, 0].imshow(latent_std, cmap='hot')
        axes[1, 0].set_title("Initial Latent (std across channels)")
        axes[1, 0].axis('off')

        # Setup timesteps for solver
        timesteps = torch.linspace(1, 0, num_sampling_steps + 1).to(device) * (num_diffusion_steps - 1)
        timesteps = (timesteps.long())[:-1]  # Exclude t=0

        intermediate_zs = []  # To store intermediate latents

        # DPM-Solver-2 algorithm
        for i in range(len(timesteps) - 1):
            # Current and next time steps
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            # Normalized time for model input
            t_normalized = t_cur / num_diffusion_steps
            if isinstance(t_normalized, torch.Tensor):
                t_normalized_tensor = t_normalized.clone().detach().to(device)
            else:
                t_normalized_tensor = torch.tensor(t_normalized, device=device)
            noise_level_scalar = (1 - torch.cos(t_normalized_tensor * math.pi / 2)).item()
            noise_level = torch.full((1, 1), noise_level_scalar, device=device)

            # Get noise prediction from model
            noise_pred = denoiser(z, noise_level, label)
            noise_pred = torch.clamp(noise_pred, -5.0, 5.0)

            # Get alpha values for current timestep
            alpha_cur = alphas_cumprod[t_cur].clamp(min=1e-5)
            alpha_next = alphas_cumprod[t_next].clamp(min=1e-5)

            # Current x0 prediction
            x0_pred = (z - torch.sqrt(1 - alpha_cur) * noise_pred) / torch.sqrt(alpha_cur)
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

            # First-order update (DPM-Solver-1)
            if i == 0:
                # For the first step, use simpler update
                z = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
            else:
                # For subsequent steps, use second-order solver
                h = t_next - t_cur
                h_last = t_cur - timesteps[i - 1]

                # Get intermediate values needed for the second-order step
                r = h_last / h

                # Calculate intermediate latent for obtaining the second noise prediction
                z_mid = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred

                # Get noise prediction at the intermediate latent
                t_next_normalized = t_next / num_diffusion_steps
                if isinstance(t_next_normalized, torch.Tensor):
                    t_next_tensor = t_next_normalized.clone().detach().to(device)
                else:
                    t_next_tensor = torch.tensor(t_next_normalized, device=device)
                noise_level_next_scalar = (1 - torch.cos(t_next_tensor * math.pi / 2)).item()
                noise_level_next = torch.full((1, 1), noise_level_next_scalar, device=device)

                noise_pred_next = denoiser(z_mid, noise_level_next, label)
                noise_pred_next = torch.clamp(noise_pred_next, -5.0, 5.0)

                # Second predicted x0
                x0_pred_next = (z_mid - torch.sqrt(1 - alpha_next) * noise_pred_next) / torch.sqrt(alpha_next)
                x0_pred_next = torch.clamp(x0_pred_next, -10.0, 10.0)

                # Apply second-order correction
                # This weighted combination of x0 predictions gives a more accurate estimate
                z = torch.sqrt(alpha_next) * ((1 + 1 / r) * x0_pred_next - (1 / r) * x0_pred) + \
                    torch.sqrt(1 - alpha_next) * noise_pred_next

                # Safety clamp
                z = torch.clamp(z, -20.0, 20.0)

            # Store intermediate result
            intermediate_zs.append(z.clone())

            # Visualize latent at this step
            latent_vis = z[0, channel_to_vis].cpu().numpy()
            axes[0, i + 1].imshow(latent_vis, cmap='viridis')
            axes[0, i + 1].set_title(f"Step {i + 1} Latent (ch {channel_to_vis})")
            axes[0, i + 1].axis('off')

            # Also show the standard deviation across channels
            latent_std = torch.std(z[0], dim=0).cpu().numpy()
            axes[1, i + 1].imshow(latent_std, cmap='hot')
            axes[1, i + 1].set_title(f"Step {i + 1} Latent (std)")
            axes[1, i + 1].axis('off')

        # Handle final step if needed
        if len(intermediate_zs) < num_sampling_steps:
            # Visualize the final latent state
            latent_vis = z[0, channel_to_vis].cpu().numpy()
            axes[0, -1].imshow(latent_vis, cmap='viridis')
            axes[0, -1].set_title(f"Final Latent (ch {channel_to_vis})")
            axes[0, -1].axis('off')

            latent_std = torch.std(z[0], dim=0).cpu().numpy()
            axes[1, -1].imshow(latent_std, cmap='hot')
            axes[1, -1].set_title(f"Final Latent (std)")
            axes[1, -1].axis('off')

        plt.tight_layout()
        output_dir = 'epoch_visualizations_latent_space'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'latent_space_denoising_class_{class_idx}.png'), dpi=200)
        plt.close()

    vae.train()
    denoiser.train()

# Modified latent diffusion model sampling visualization with DDIM
def generate_latent_diffusion_samples(latent_model, epoch, num_diffusion_steps, alphas_cumprod, betas):
    vae = latent_model.vae
    denoiser = latent_model.denoiser
    latent_spatial_size = 16

    vae.eval()
    denoiser.eval()

    # DPM-Solver-2 sampling parameters
    num_sampling_steps = 50  # Much fewer steps than DDIM (50)

    with torch.no_grad():
        # Generate samples for each class
        fig, axes = plt.subplots(5, 10, figsize=(20, 10))

        for class_idx in range(10):
            # Generate 5 samples at once for efficiency
            batch_size = 5
            class_labels = torch.tensor([class_idx] * batch_size, device=device)

            # Use DPM-Solver-2 to generate images
            images, _ = dpm_solver_2_sampling(
                latent_model,
                num_diffusion_steps,
                alphas_cumprod,
                class_labels=class_labels,
                batch_size=batch_size,
                num_sampling_steps=num_sampling_steps
            )

            # Process and display the generated images
            for sample_idx in range(5):
                img = images[sample_idx].permute(1, 2, 0).squeeze().numpy()

                # Replace any NaNs in final image
                if np.isnan(img).any():
                    img = np.nan_to_num(img, nan=0.5)

                # Display image
                axes[sample_idx, class_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[sample_idx, class_idx].axis('off')

                if sample_idx == 0:
                    axes[sample_idx, class_idx].set_title(f"{class_names[class_idx]}")

        plt.tight_layout()
        output_dir = 'epoch_visualizations_latent'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'latent_diffusion_samples_epoch_{epoch + 1}_dpm.png'), dpi=200)
        plt.close()

    vae.train()
    denoiser.train()


# Define diffusion timesteps
num_diffusion_steps = 20

# Main function
if __name__ == '__main__':
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Current device: {device}")

    # Create VAE model
    latent_dim = 32  # Latent space dimension
    vae = VAE(in_channels=1, latent_dim=latent_dim).to(device)

    # Create latent space U-Net denoiser
    denoiser = LatentUNetDenoiser(
        latent_dim=latent_dim,  # Corresponds to latent space dimension
        num_classes=10,  # Fashion MNIST has 10 classes
        time_dim=256,
        attention_heads=8
    ).to(device)

    # Create latent diffusion model
    latent_model = LatentDiffusionModel(vae, denoiser, latent_dim=latent_dim).to(device)

    # Create output directory
    os.makedirs('latent_diffusion_results', exist_ok=True)

    # Step 1: Train VAE
    vae_model_path = 'fashion_mnist_vae_final.pt'
    if os.path.exists(vae_model_path):
        print("Found existing VAE parameters, loading them...")
        vae.load_state_dict(torch.load(vae_model_path, map_location=device))
        vae.eval()
    else:
        # Do train if no parameters preserved
        vae, losses, recon_losses, kld_losses = train_vae(vae, num_epochs=50)
        torch.save(vae.state_dict(), vae_model_path)

    # Step 2: Train latent diffusion model with EMA
    denoiser, ema_denoiser = train_latent_diffusion(vae, denoiser, latent_model,
                                                    num_epochs=200,
                                                    num_diffusion_steps=num_diffusion_steps)

    # Save both regular and EMA models
    torch.save({
        'vae': vae.state_dict(),
        'denoiser': denoiser.state_dict(),
        'ema_denoiser': ema_denoiser.state_dict()
    }, 'fashion_mnist_latent_diffusion_final.pt')

    print("Training complete!")
