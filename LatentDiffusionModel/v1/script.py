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

# Set image size for Fashion MNIST (28x28 grayscale images)
img_size = 28

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # This already scales to [0,1] range
])

# Load Fashion MNIST dataset
batch_size = 256  # Increased for more stable gradients
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
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
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
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()  # Enforce output range [0,1] for stability
        )

        self.latent_dim = latent_dim
        # Control parameter for KL weight
        self.kl_weight = 0.0001  # Very small initial KL weight

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


# Self-attention module
class SelfAttention(nn.Module):
    """
    Linear attention module with O(n) complexity instead of O(n²)
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


# Latent space U-Net denoiser for diffusion model
class LatentUNetDenoiser(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10, time_dim=256, attention_heads=8):
        super(LatentUNetDenoiser, self).__init__()

        # Store important parameters
        self.latent_dim = latent_dim
        self.time_dim = time_dim

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

        # Initial convolutional layer
        self.init_conv = DoubleConv(latent_dim, 64)

        # Encoder (downsampling path)
        self.down1 = Down(64, 192, emb_dim=time_dim)
        self.sa1 = SelfAttention(192, num_heads=attention_heads)
        self.down2 = Down(192, 384, emb_dim=time_dim)
        self.sa2 = SelfAttention(384, num_heads=attention_heads)
        self.down3 = Down(384, 384, emb_dim=time_dim)
        self.sa3 = SelfAttention(384, num_heads=attention_heads)

        # Bottleneck layers
        self.bottleneck1 = DoubleConv(384, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 384)

        # Decoder (upsampling path) - with correct channel specifications
        # Parameters: (input_channels, skip_channels, output_channels)
        self.up1 = Up(384, 384, 192, emb_dim=time_dim)  # Fixed: takes 384 from bottleneck, 384 from z3, outputs 192
        self.sa4 = SelfAttention(192, num_heads=attention_heads)
        self.up2 = Up(192, 192, 64, emb_dim=time_dim)  # Fixed: takes 192 from up1, 192 from z2, outputs 64
        self.sa5 = SelfAttention(64, num_heads=attention_heads)
        self.up3 = Up(64, 64, 64, emb_dim=time_dim)  # Fixed: takes 64 from up2, 64 from z1, outputs 64
        self.sa6 = SelfAttention(64, num_heads=attention_heads)

        # Final output convolution
        self.final_conv = nn.Sequential(
            DoubleConv(64, 64),
            nn.Conv2d(64, latent_dim, kernel_size=1)  # 1x1 conv to map back to latent dimension
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
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

    def sample_from_latent(self, x):
        with torch.no_grad():
            mu, log_var = self.vae.encode(x)
            z = self.vae.reparameterize(mu, log_var)
            latent_spatial_size = 16
            z = z.view(z.size(0), z.size(1), 1, 1)
            z = F.interpolate(z, size=(latent_spatial_size, latent_spatial_size), mode='bilinear', align_corners=True)
        return z

    def decode(self, z):
        # Decode latent representation
        with torch.no_grad():
            reconstruction = self.vae.decode(z)
        return reconstruction

    def forward(self, x, noise_level, labels):
        # Diffusion prediction in latent space
        return self.denoiser(x, noise_level, labels)


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

    # Much simpler KL annealing - linear warmup over first half of training
    def get_kl_weight(epoch, num_epochs):
        warmup_epochs = num_epochs // 2
        if epoch < warmup_epochs:
            return 0.0001 + (0.001 - 0.0001) * (epoch / warmup_epochs)
        else:
            return 0.001  # Fixed small weight after warmup

    # FIXED: Using MSE loss which is guaranteed to be positive
    def reconstruction_loss(recon_x, x):
        """
        Simple MSE loss that's guaranteed to be positive
        """
        return F.mse_loss(recon_x, x, reduction='mean')

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


# Train latent diffusion model
def train_latent_diffusion(vae, denoiser, latent_model, num_epochs=200, num_diffusion_steps=100):
    print("Starting latent diffusion model training...")

    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    optimizer = optim.AdamW(denoiser.parameters(), lr=1e-4, weight_decay=0.01, eps=1e-8)
    mse_loss = nn.MSELoss()

    # Learning rate scheduler
    def warmup_cosine_schedule(step, warmup_steps, max_steps):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Training parameters
    max_steps = num_epochs * len(train_loader)
    warmup_steps = max_steps // 10

    # Diffusion process parameters
    betas = torch.linspace(1e-5, 0.01, num_diffusion_steps).to(device)
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

    training_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        denoiser.train()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Use modified sampling method, preserving the variational properties of VAE
            z = latent_model.sample_from_latent(images)
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(f"Warning: NaN or Inf in latent sample at epoch {epoch}, step {step}")
                z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
            z = torch.clamp(z, min=-10.0, max=10.0)

            # Randomly select timestep
            max_t = min(int((epoch / num_epochs) * num_diffusion_steps) + 10, num_diffusion_steps)
            t = torch.randint(1, max_t + 1, (z.shape[0],), device=device)

            # Add noise
            noisy_z, target_noise = q_sample(z, t - 1)

            # Normalize timestep - using non-linear transformation for better time representation
            noise_level = (1 - torch.cos((t - 1) / num_diffusion_steps * math.pi / 2)).view(-1, 1)

            optimizer.zero_grad()

            # Predict noise
            predicted_noise = denoiser(noisy_z, noise_level, labels)

            if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                print(f"Warning: NaN in predicted noise at epoch {epoch}, step {step}")
                predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=1.0, neginf=-1.0)
                continue

            if torch.isnan(target_noise).any() or torch.isinf(target_noise).any():
                print(f"Warning: NaN in target noise at epoch {epoch}, step {step}")
                target_noise = torch.nan_to_num(target_noise, nan=0.0, posinf=1.0, neginf=-1.0)
                continue

            loss = mse_loss(predicted_noise, target_noise)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN loss at epoch {epoch}, step {step}")
                continue  # skip this batch

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=0.5)

            # Update learning rate
            global_step = epoch * len(train_loader) + step
            lr_scale = warmup_cosine_schedule(global_step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * 2e-4

            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f'Latent Diffusion Training: Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        print(f'Latent Diffusion Training: Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save(denoiser.state_dict(), f'fashion_mnist_latent_diffusion_epoch_{epoch + 1}.pt')

        # Save generated visualizations
        if (epoch + 1) % 1 == 0:
            generate_latent_diffusion_samples(latent_model, epoch, num_diffusion_steps, alphas_cumprod, betas)

    print("Latent diffusion model training completed!")
    return denoiser


# Latent diffusion model sampling visualization with DDIM
def generate_latent_diffusion_samples(latent_model, epoch, num_diffusion_steps, alphas_cumprod, betas):
    vae = latent_model.vae
    denoiser = latent_model.denoiser
    latent_spatial_size = 16

    vae.eval()
    denoiser.eval()

    # DDIM sampling parameters
    eta = 0.0  # Deterministic sampling for clarity
    skip = num_diffusion_steps // 50
    timesteps = list(range(0, num_diffusion_steps, skip))
    timesteps.reverse()  # Reversed order from T to 0

    with torch.no_grad():
        # Generate samples for each class
        fig, axes = plt.subplots(5, 10, figsize=(20, 10))

        for class_idx in range(10):
            for sample_idx in range(5):
                # Sample a latent vector from standard normal distribution
                z = torch.randn((1, latent_model.latent_dim, latent_spatial_size, latent_spatial_size)).to(device)
                # Ensure no NaNs in initial noise
                if torch.isnan(z).any():
                    z = torch.zeros_like(z) + torch.randn_like(z) * 0.1

                label = torch.tensor([class_idx], device=device)
                successful_sample = True  # Flag to track if sampling completes successfully

                # DDIM reverse diffusion process with safeguards
                # DDIM reverse diffusion process with enhanced safeguards for final steps
                for i in range(len(timesteps) - 1):
                    t_current = timesteps[i]
                    t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0

                    # Use more conservative approach in final steps
                    final_steps = (i >= len(timesteps) - 10)  # Last 10 steps

                    # Current timestep noise level
                    noise_level = (1 - torch.cos(
                        torch.tensor([t_current], device=device) / num_diffusion_steps * math.pi / 2)).view(-1, 1)

                    try:
                        # Predict noise
                        noise_pred = denoiser(z, noise_level, label)

                        # More aggressive denoising for final steps
                        if torch.isnan(noise_pred).any():
                            print(f"Noise prediction contains NaNs at step {i}")
                            noise_pred = torch.zeros_like(noise_pred)  # Use zeros instead of random noise
                        else:
                            # Apply stricter clamping in final steps
                            noise_pred = noise_pred.clamp(-5.0, 5.0) if final_steps else noise_pred.clamp(-10.0, 10.0)

                        # Safely extract coefficients with larger minimum bounds for final steps
                        min_alpha = 1e-4 if final_steps else 1e-5
                        alpha_current = alphas_cumprod[t_current].clamp(min=min_alpha)
                        alpha_current_sqrt = torch.sqrt(alpha_current).clamp(min=min_alpha)
                        sigma_current = torch.sqrt((1 - alpha_current).clamp(min=0))

                        # In final steps, use simpler update rule
                        if final_steps:
                            # For final steps, directly blend toward mean
                            x0_pred = (z - sigma_current * noise_pred) / alpha_current_sqrt

                            # Extra safeguards for final steps
                            x0_pred = torch.clamp(x0_pred, -20.0, 20.0)

                            if t_next > 0:
                                alpha_next = alphas_cumprod[t_next].clamp(min=min_alpha)
                                alpha_next_sqrt = torch.sqrt(alpha_next).clamp(min=min_alpha)

                                # Simplified update for final steps - direct interpolation
                                z_next = alpha_next_sqrt * x0_pred

                                # Add minimal noise in final steps
                                z_next = z_next + torch.randn_like(z_next) * 0.01
                            else:
                                # Last step - directly use prediction
                                z = x0_pred
                        else:
                            # Normal steps use regular DDIM
                            x0_pred = (z - sigma_current * noise_pred) / alpha_current_sqrt

                            if torch.isnan(x0_pred).any():
                                x0_pred = torch.nan_to_num(x0_pred, nan=0.0)
                                x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

                            if t_next > 0:
                                alpha_next = alphas_cumprod[t_next].clamp(min=1e-5)
                                alpha_next_sqrt = torch.sqrt(alpha_next).clamp(min=1e-5)

                                # Safer direction calculation
                                variance = 0.0  # Set to 0 to simplify the process
                                sqrt_term = (1 - alpha_next).clamp(min=1e-10)
                                direction = torch.sqrt(sqrt_term) * noise_pred

                                if torch.isnan(direction).any():
                                    direction = torch.zeros_like(direction)

                                # DDIM update
                                z_next = alpha_next_sqrt * x0_pred + direction
                            else:
                                z = x0_pred

                        # Final NaN check on new z
                        if t_next > 0:
                            if torch.isnan(z_next).any():
                                print(f"New z contains NaNs at step {i}, using recovery")
                                # More aggressive recovery
                                z_next = x0_pred if not torch.isnan(x0_pred).any() else z

                            # Additional clipping for stability
                            z_next = torch.clamp(z_next, -50.0, 50.0)
                            z = z_next

                    except Exception as e:
                        print(f"Error in diffusion step {i}: {e}")
                        successful_sample = False
                        break

                try:
                    if successful_sample:
                        # Convert spatial latent to vector form the VAE can use
                        # Use a safer conversion - center crop approach
                        z_center = z[:, :, z.size(2) // 2 - 1:z.size(2) // 2 + 1, z.size(3) // 2 - 1:z.size(3) // 2 + 1]
                        z_flat = z_center.mean(dim=[2, 3])  # Mean of center 2x2 patch

                        # Final NaN check before decoding
                        if torch.isnan(z_flat).any():
                            z_flat = torch.zeros_like(z_flat) + torch.randn_like(z_flat) * 0.01

                        # Decode latent representation to image
                        img = vae.decode(z_flat)
                        img = img.cpu().permute(0, 2, 3, 1).squeeze().numpy()

                        # Replace any NaNs in final image
                        if np.isnan(img).any():
                            img = np.nan_to_num(img, nan=0.5)
                    else:
                        # If sampling failed, create a gray image
                        img = np.ones((28, 28)) * 0.5
                except Exception as e:
                    print(f"Error in final decoding: {e}")
                    img = np.ones((28, 28)) * 0.5

                # Display image
                axes[sample_idx, class_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[sample_idx, class_idx].axis('off')

                if sample_idx == 0:
                    axes[sample_idx, class_idx].set_title(f"{class_names[class_idx]}")

        plt.tight_layout()
        output_dir = 'epoch_visualizations_latent'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'latent_diffusion_samples_epoch_{epoch + 1}_ddim.png'), dpi=200)
        plt.close()

    vae.train()
    denoiser.train()


# Define diffusion timesteps
num_diffusion_steps = 50

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

    # Step 2: Train latent diffusion model
    denoiser = train_latent_diffusion(vae, denoiser, latent_model, num_epochs=200,
                                      num_diffusion_steps=num_diffusion_steps)

    # Save trained latent diffusion model
    torch.save({
        'vae': vae.state_dict(),
        'denoiser': denoiser.state_dict()
    }, 'fashion_mnist_latent_diffusion_final.pt')

    print("Training complete!")
