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
from tqdm.auto import tqdm
from sklearn.manifold import TSNE

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


def euclidean_distance_loss(x, y, reduction='mean'):
    """
    Calculate the Euclidean distance between x and y tensors.

    Args:
        x: First tensor
        y: Second tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Euclidean distance loss
    """
    # Calculate squared differences
    squared_diff = (x - y) ** 2

    # Sum across all dimensions except batch
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)

    # Take square root to get Euclidean distance
    euclidean_dist = torch.sqrt(squared_dist + 1e-8)  # Add small epsilon to avoid numerical instability

    # Apply reduction
    if reduction == 'mean':
        return euclidean_dist.mean()
    elif reduction == 'sum':
        return euclidean_dist.sum()
    else:  # 'none'
        return euclidean_dist


# Channel Attention Layer for improved feature selection
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # Global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Convolutional Attention Block - building block for encoder/decoder
class CAB(nn.Module):
    def __init__(self, n_feat, reduction=16, bias=False):
        super(CAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias),
        )
        self.ca = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return res + x  # Residual Connection


# Encoder network with CAB blocks
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, in_channels=1):
        super().__init__()

        # Encoder channel progression
        self.encoder = nn.Sequential(
            # First stage: in_channels -> 16 channels
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            CAB(16, reduction=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, stride=2, padding=1),  # 28x28 -> 14x14

            # Second stage: 16 -> 32 channels
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            CAB(32, reduction=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # 14x14 -> 7x7
        )

        # Flatten and project to latent space
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.flatten(x)


# Decoder network with CAB blocks
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()

        # Initial linear transformation from latent space
        self.linear = nn.Linear(latent_dim, 32 * 7 * 7)

        # Decoder channel progression
        self.decoder = nn.Sequential(
            # First stage: 32 -> 16 channels
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 7x7 -> 14x14
            CAB(16, reduction=8),
            nn.ReLU(inplace=True),

            # Second stage: 16 -> out_channels
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output activation for [0,1] range
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 32, 7, 7)  # Reshape to spatial dimensions
        return self.decoder(x)


# Complete Autoencoder model
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, in_channels=1):
        super().__init__()
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


# Swish activation function for UNet
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Time embedding for diffusion model
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        # Sinusoidal time embedding similar to positional encoding
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Process through MLP
        return self.lin2(self.act(self.lin1(emb)))


# Residual block for UNet
class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=16, num_groups=8, dropout_rate=0.2):
        super().__init__()

        # Feature normalization and convolution
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb = nn.Linear(d_time, out_channels)
        self.act = Swish()

        self.dropout = nn.Dropout(dropout_rate)

        # Second convolution
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        # First part
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        t_emb = self.act(self.time_emb(t))
        h = h + t_emb.view(-1, t_emb.shape[1], 1, 1)

        # Second part
        h = self.act(self.norm2(h))

        h = self.dropout(h)

        h = self.conv2(h)

        # Residual connection
        return h + self.residual(x)


# Attention block for UNet
class UNetAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        # Normalize input
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape for attention computation
        q = q.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]
        k = k.permute(0, 1, 2, 3)  # [b, heads, c//heads, h*w]
        v = v.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]

        # Compute attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.matmul(attn, v)  # [b, heads, h*w, c//heads]
        out = out.permute(0, 3, 1, 2)  # [b, c//heads, heads, h*w]
        out = out.reshape(b, c, h, w)

        # Project and add residual
        return self.proj(out) + residual


# Switch Sequential for handling time embeddings
class SwitchSequential(nn.Sequential):
    def forward(self, x, t=None):
        for layer in self:
            if isinstance(layer, (UNetResidualBlock, UNetAttentionBlock)):
                x = layer(x, t)
            else:
                x = layer(x)
        return x


# UNet architecture for noise prediction
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[16, 32, 64], dropout_rate=0.2):
        super().__init__()

        # Time embedding
        self.time_emb = TimeEmbedding(n_channels=16)

        # Downsampling path (encoder)
        self.down_blocks = nn.ModuleList()

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

        # Downsampling blocks
        input_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.down_blocks.append(
                nn.ModuleList([
                    UNetResidualBlock(input_dim, input_dim),
                    UNetResidualBlock(input_dim, input_dim),
                    nn.Conv2d(input_dim, dim, 4, stride=2, padding=1)  # Downsample
                ])
            )
            input_dim = dim

        self.dropout_mid = nn.Dropout(dropout_rate)

        # Middle block (bottleneck)
        self.middle_blocks = nn.ModuleList([
            UNetResidualBlock(hidden_dims[-1], hidden_dims[-1]),
            UNetAttentionBlock(hidden_dims[-1]),
            UNetResidualBlock(hidden_dims[-1], hidden_dims[-1])
        ])

        # Upsampling path (decoder)
        self.up_blocks = nn.ModuleList()

        # Upsampling blocks - FIX: Reorder modules and update channel dimensions
        for dim in reversed(hidden_dims[:-1]):
            self.up_blocks.append(
                nn.ModuleList([
                    # Use hidden_dims[-1] as input and output for upsampling
                    nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 4, stride=2, padding=1),
                    # Handle concatenated channels from skip connection
                    UNetResidualBlock(hidden_dims[-1] + dim, dim),
                    UNetResidualBlock(dim, dim),
                ])
            )
            hidden_dims[-1] = dim

        self.dropout_final = nn.Dropout(dropout_rate)

        # Final blocks - Handle concatenated input
        self.final_block = SwitchSequential(
            UNetResidualBlock(hidden_dims[0] * 2, hidden_dims[0]),
            nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_emb(t)

        # Initial convolution
        x = self.initial_conv(x)

        # Store skip connections
        skip_connections = [x]

        # Downsampling
        for resblock1, resblock2, downsample in self.down_blocks:
            x = resblock1(x, t_emb)
            x = resblock2(x, t_emb)
            skip_connections.append(x)
            x = downsample(x)

        x = self.dropout_mid(x)

        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, UNetAttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)

        # Upsampling - FIX: Match the new module order
        for upsample, resblock1, resblock2 in self.up_blocks:
            x = upsample(x)  # Upsample first
            x = torch.cat([x, skip_connections.pop()], dim=1)  # Then concatenate
            x = resblock1(x, t_emb)  # Then process
            x = resblock2(x, t_emb)

        x = self.dropout_final(x)

        # Final blocks
        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_block(x, t_emb)

        return x


# Simplified denoising diffusion model
class SimpleDenoiseDiffusion():
    def __init__(self, eps_model, n_steps=1000, device=None):
        super().__init__()
        self.eps_model = eps_model
        self.device = device

        # Linear beta schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def q_sample(self, x0, t, eps=None):
        """Forward diffusion process: add noise to data"""
        if eps is None:
            eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    def p_sample(self, xt, t):
        """Single denoising step"""
        # Convert time to tensor format expected by model
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)

        # Predict noise
        eps_theta = self.eps_model(xt, t)

        # Get alpha values
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)

        # Calculate mean
        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_t)

        # Add noise if not the final step
        var = self.beta[t].reshape(-1, 1, 1, 1)

        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    def sample(self, shape, device):
        """Generate samples by denoising from pure noise"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Progressively denoise
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t)

        return x

    def loss(self, x0):
        """Calculate noise prediction loss"""
        batch_size = x0.shape[0]

        # Random timestep for each sample
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Add noise
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)

        # Predict noise
        eps_theta = self.eps_model(xt, t)

        return euclidean_distance_loss(eps, eps_theta)


# Visualization functions

# Visualize autoencoder reconstructions
def visualize_reconstructions(autoencoder, epoch, save_dir="./results"):
    """Visualize original and reconstructed images at each epoch"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Get a batch of test data
    test_loader = DataLoader(
        datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=8, shuffle=True
    )

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # Generate reconstructions
    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(test_images)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Original
        img = test_images[i].cpu().squeeze().numpy()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original: {class_names[test_labels[i]]}')
        axes[0, i].axis('off')

        # Reconstruction
        recon_img = reconstructed[i].cpu().squeeze().numpy()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_epoch_{epoch}.png")
    plt.close()
    autoencoder.train()


# Visualize latent space with t-SNE
def visualize_latent_space(autoencoder, epoch, save_dir="./results"):
    """Visualize the latent space of the autoencoder using t-SNE"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Get test data
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)  # Use a larger batch for visualization

    # Extract features and labels
    autoencoder.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            latents = autoencoder.encode(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    # Combine batches
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)

    # Use t-SNE for dimensionality reduction
    try:
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(all_latents)

        # Plot the 2D latent space
        plt.figure(figsize=(10, 8))
        for i in range(10):  # 10 classes
            mask = all_labels == i
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], label=class_names[i], alpha=0.6)

        plt.title(f"t-SNE Visualization of Latent Space (Epoch {epoch})")
        plt.legend()
        plt.savefig(f"{save_dir}/latent_space_epoch_{epoch}.png")
        plt.close()
    except ImportError:
        print("sklearn not installed, skipping t-SNE visualization")

    autoencoder.train()


# Visualize latent space interpolation
def visualize_latent_interpolation(autoencoder, epoch, save_dir="./results"):
    """Visualize interpolation between two random samples in latent space"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Get two random samples from test set
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    idx1, idx2 = np.random.randint(0, len(test_dataset), 2)
    img1, label1 = test_dataset[idx1]
    img2, label2 = test_dataset[idx2]

    # Encode images to latent space
    autoencoder.eval()
    with torch.no_grad():
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        latent1 = autoencoder.encode(img1)
        latent2 = autoencoder.encode(img2)

        # Create interpolated points
        steps = 10
        interpolated_latents = []
        for alpha in np.linspace(0, 1, steps):
            interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
            interpolated_latents.append(interpolated_latent)

        # Decode interpolated points
        interpolated_images = []
        for latent in interpolated_latents:
            decoded = autoencoder.decode(latent)
            interpolated_images.append(decoded)

        # Visualize
        fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 3))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img.cpu().squeeze().numpy(), cmap='gray')
            axes[i].axis('off')

        fig.suptitle(f"Interpolation: {class_names[label1]} → {class_names[label2]}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/interpolation_epoch_{epoch}.png")
        plt.close()

    autoencoder.train()


# Visualize latent space denoising process
def visualize_denoising_steps(autoencoder, diffusion, epoch, class_idx=None, save_dir="./results"):
    """
    Visualize both the denoising process and the corresponding path in latent space.

    Args:
        autoencoder: Trained autoencoder model
        diffusion: Trained diffusion model
        epoch: Current epoch number
        class_idx: Target class index (0-9)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    # ===== PART 1: Setup dimensionality reduction for latent space =====
    print(f"Generating latent space projection...")
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Extract features and labels
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            latents = autoencoder.encode(images)
            all_latents.append(latents.detach().cpu().numpy())
            all_labels.append(labels.numpy())

    # Combine batches
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)

    # Use PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(all_latents)

    # ===== PART 2: Setup denoising visualization =====
    # Parameters for visualization
    n_samples = 5  # Reduced from 10 to make visualization clearer
    steps_to_show = 8
    step_size = diffusion.n_steps // steps_to_show
    timesteps = list(range(0, diffusion.n_steps, step_size))[::-1]

    # Generate sample from pure noise
    x = torch.randn((n_samples, 1, 8, 8), device=device)

    # Store denoised samples at each timestep
    samples_per_step = []
    # Track latent path for the first sample
    path_latents = []

    # ===== PART 3: Perform denoising and track path =====
    with torch.no_grad():
        for t in timesteps:
            # Current denoised state
            current_x = x.clone()

            # Denoise from current step to t=0
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device))

            # Store the latent vector for the first sample for path visualization
            path_latents.append(current_x[0:1].view(1, -1).detach().cpu().numpy())

            # Decode to images
            current_x_flat = current_x.view(n_samples, -1)
            decoded = autoencoder.decode(current_x_flat)

            # Add to samples
            samples_per_step.append(decoded.cpu())

        # Add final denoised state to path
        path_latents.append(current_x[0:1].view(1, -1).detach().cpu().numpy())

    # Stack path latents
    path_latents = np.vstack(path_latents)

    # Project path points to PCA space
    path_2d = pca.transform(path_latents)

    # ===== PART 4: Create combined visualization =====
    # Create a figure with 2 subplots: denoising process (top) and latent path (bottom)
    fig = plt.figure(figsize=(16, 16))

    # Configure subplot layout
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)

    # ===== PART 5: Plot denoising process (top subplot) =====
    ax_denoising = fig.add_subplot(gs[0])

    # Create a grid for the denoising visualization
    grid_rows = n_samples
    grid_cols = len(timesteps)

    # Create a class description
    class_description = f"class: {class_names[class_idx]}" if class_idx is not None else "random samples"

    # Set title for the denoising subplot
    ax_denoising.set_title(f"Diffusion Model Denoising Process (Epoch {epoch}, {class_description})",
                           fontsize=16, pad=10)

    # Hide axis ticks
    ax_denoising.set_xticks([])
    ax_denoising.set_yticks([])

    # Create nested gridspec for the denoising images
    gs_denoising = gs[0].subgridspec(grid_rows, grid_cols, wspace=0.1, hspace=0.1)

    # Plot each denoising step
    for i in range(n_samples):
        for j, t in enumerate(timesteps):
            ax = fig.add_subplot(gs_denoising[i, j])
            img = samples_per_step[j][i].squeeze().numpy()
            ax.imshow(img, cmap='gray')

            # Add timestep labels only to the top row
            if i == 0:
                ax.set_title(f't={t}', fontsize=9)

            # Add sample labels only to the leftmost column
            if j == 0:
                ax.set_ylabel(f"Sample {i + 1}", fontsize=9)

            # Highlight the first sample that corresponds to the path
            if i == 0:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add text indicating the first row corresponds to the latent path
    plt.figtext(0.02, 0.65, "Path Tracked →", fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # ===== PART 6: Plot latent space path (bottom subplot) =====
    ax_latent = fig.add_subplot(gs[1])

    # Plot each class with alpha transparency
    for i in range(10):
        mask = all_labels == i
        alpha = 0.3 if i != class_idx else 0.8  # Highlight target class
        size = 20 if i != class_idx else 40  # Larger points for target class
        ax_latent.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            label=class_names[i],
            alpha=alpha,
            s=size
        )

    # Plot the diffusion path
    ax_latent.plot(
        path_2d[:, 0],
        path_2d[:, 1],
        'r-o',
        linewidth=2.5,
        markersize=8,
        label=f"Diffusion Path",
        zorder=10  # Ensure path is drawn on top
    )

    # Add arrows to show direction
    for i in range(len(path_2d) - 1):
        ax_latent.annotate(
            "",
            xy=(path_2d[i + 1, 0], path_2d[i + 1, 1]),
            xytext=(path_2d[i, 0], path_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5)
        )

    # Add timestep labels along the path
    for i, t in enumerate(timesteps):
        ax_latent.annotate(
            f"t={t}",
            xy=(path_2d[i, 0], path_2d[i, 1]),
            xytext=(path_2d[i, 0] + 2, path_2d[i, 1] + 2),
            fontsize=8,
            color='darkred'
        )

    # Add markers for start and end points
    ax_latent.scatter(path_2d[0, 0], path_2d[0, 1], c='black', s=100, marker='x', label="Start (Noise)", zorder=11)
    ax_latent.scatter(path_2d[-1, 0], path_2d[-1, 1], c='green', s=100, marker='*', label="End (Generated)",
                      zorder=11)

    # Highlight target class area if specified
    if class_idx is not None:
        target_mask = all_labels == class_idx
        target_center = np.mean(latents_2d[target_mask], axis=0)
        ax_latent.scatter(target_center[0], target_center[1], c='green', s=300, marker='*',
                          edgecolor='black', alpha=0.7, zorder=9)
        ax_latent.annotate(
            f"TARGET: {class_names[class_idx]}",
            xy=(target_center[0], target_center[1]),
            xytext=(target_center[0] + 5, target_center[1] + 5),
            fontsize=14,
            fontweight='bold',
            color='darkgreen',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
        )

    ax_latent.set_title(f"Diffusion Path in Latent Space (Epoch {epoch})", fontsize=16)
    ax_latent.legend(fontsize=10, loc='best')
    ax_latent.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the denoising process (top) and the corresponding path in latent space (bottom).\n"
        "The first row of images (highlighted in red) corresponds to the red path in the latent space plot below.",
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Create safe filename
    title = f"denoising_with_path_epoch_{epoch}"
    if class_idx is not None:
        # Replace forward slash with hyphen to avoid file path issues
        safe_class_name = class_names[class_idx].replace('/', '-')
        title += f"_class_{safe_class_name}"

    # Save the figure
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    plt.savefig(f"{save_dir}/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined visualization saved to {save_dir}/{title}.png")

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()


# Generate samples for all classes
def generate_samples_grid(autoencoder, diffusion, epoch, n_per_class=5, save_dir="./results"):
    """Generate a grid of samples with n_per_class samples for each Fashion MNIST class"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    n_classes = 10
    # Create figure with extra column for class labels
    fig, axes = plt.subplots(n_classes, n_per_class + 1, figsize=((n_per_class + 1) * 2, n_classes * 2))

    # Add a title to explain what the figure shows
    fig.suptitle(f'Fashion MNIST Samples Generated by Diffusion Model (Epoch {epoch})',
                 fontsize=16, y=0.98)

    for i in range(n_classes):
        # Create a text-only cell for the class name
        axes[i, 0].text(0.5, 0.5, class_names[i],
                        fontsize=14, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        axes[i, 0].axis('off')

        # Generate samples
        latent_shape = (n_per_class, 1, 8, 8)
        samples = diffusion.sample(latent_shape, device)

        # Decode samples
        with torch.no_grad():
            samples_flat = samples.view(n_per_class, -1)
            decoded = autoencoder.decode(samples_flat)

        # Plot samples (starting from column 1, as column 0 is for class names)
        for j in range(n_per_class):
            img = decoded[j].cpu().squeeze().numpy()
            axes[i, j + 1].imshow(img, cmap='gray')

            # Remove axis ticks
            axes[i, j + 1].axis('off')

            # Add sample numbers above the first row
            if i == 0:
                axes[i, j + 1].set_title(f'Sample {j + 1}', fontsize=9)

    # Add a text box explaining the visualization
    description = (
        "This visualization shows fashion items generated by the diffusion model.\n"
        "The model creates new, synthetic images based on learned patterns from Fashion MNIST.\n"
        "Each row corresponds to a different clothing category as labeled."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to make room for titles
    plt.savefig(f"{save_dir}/samples_grid_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    print(f"Generated sample grid for epoch {epoch} with clearly labeled fashion categories")

# Modified training function for the autoencoder with enhanced visualizations
def train_autoencoder(autoencoder, num_epochs=80, lr=1e-4, visualize_every=5, save_dir="./results"):
    print("Starting Autoencoder training...")
    os.makedirs(save_dir, exist_ok=True)

    device = next(autoencoder.parameters()).device
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    def criterion(x, y):
        return euclidean_distance_loss(x, y)

    # Training loop
    loss_history = []

    # Create a figure for live loss plotting
    plt.figure(figsize=(10, 5))

    for epoch in range(num_epochs):
        epoch_loss = 0
        autoencoder.train()

        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            data = data.to(device)

            # Forward pass
            reconstructed = autoencoder(data)
            loss = criterion(reconstructed, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Visualizations
        if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
            # Reconstruction visualization
            visualize_reconstructions(autoencoder, epoch + 1, save_dir)

            # Latent space visualization
            visualize_latent_space(autoencoder, epoch + 1, save_dir)

            # Latent space interpolation
            visualize_latent_interpolation(autoencoder, epoch + 1, save_dir)

            # Plot and save the current loss curve
            plt.clf()
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
            plt.title('Autoencoder Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f"{save_dir}/autoencoder_loss_progress.png")

            # Save checkpoint
            torch.save(autoencoder.state_dict(), f"{save_dir}/autoencoder_epoch_{epoch + 1}.pt")

    plt.close()
    return autoencoder, loss_history


# Training function for the diffusion model
def train_diffusion(autoencoder, unet, num_epochs=100, lr=1e-3, visualize_every=10, save_dir="./results"):
    """
    Train the diffusion model with dynamic batch size adjustment.

    Args:
        autoencoder: Trained autoencoder model
        unet: UNet model for noise prediction
        num_epochs: Number of training epochs
        lr: Initial learning rate
        visualize_every: Epoch interval for visualizations
        save_dir: Directory to save results

    Returns:
        unet: Trained UNet model
        diffusion: Trained diffusion model
        loss_history: List of average losses per epoch
    """
    print("Starting Diffusion Model training...")
    os.makedirs(save_dir, exist_ok=True)

    device = next(autoencoder.parameters()).device
    autoencoder.eval()  # Set autoencoder to evaluation mode

    # Get reference to the training dataset
    train_dataset = train_loader.dataset

    # Initialize batch size tracking
    current_batch_size = batch_size  # Use the global batch size to start
    current_loader = train_loader

    # Create diffusion model
    diffusion = SimpleDenoiseDiffusion(unet, n_steps=1000, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # For early stopping
    best_loss = float('inf')
    patience = 15
    counter = 0
    best_model_path = f"{save_dir}/best_diffusion_model.pt"

    # Training loop
    loss_history = []

    # Create a figure for live loss plotting
    plt.figure(figsize=(10, 5))

    for epoch in range(num_epochs):
        # Dynamic batch size adjustment logic
        if epoch == 50:  # When reaching the stagnation point
            new_batch_size = current_batch_size // 2  # Halve the batch size
            print(f"\nReducing batch size from {current_batch_size} to {new_batch_size} at epoch {epoch + 1}")
            current_batch_size = new_batch_size

            # Create a new data loader with the smaller batch size
            current_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

        # Alternatively, use loss plateau detection for dynamic adjustment
        elif epoch > 30 and len(loss_history) >= 3:  # At least 3 epochs of history
            # If little improvement in the last 3 epochs
            if loss_history[-3] - loss_history[-1] < 0.0005:
                # And current batch size is above some minimum
                if current_batch_size > 64:
                    new_batch_size = current_batch_size // 2
                    print(f"\nLoss plateau detected. Reducing batch size from {current_batch_size} to {new_batch_size}")
                    current_batch_size = new_batch_size
                    current_loader = DataLoader(
                        train_dataset,
                        batch_size=current_batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True
                    )

        epoch_loss = 0
        # Use the current data loader
        for batch_idx, (data, _) in enumerate(tqdm(current_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            data = data.to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = autoencoder.encode(data)
                # Reshape latents to spatial form [B, 1, 8, 8]
                latents = latents.view(-1, 1, 8, 8)

            # Calculate diffusion loss
            loss = diffusion.loss(latents)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.6f}")

        # Calculate average loss considering the new number of batches
        avg_loss = epoch_loss / len(current_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            # Save the best model
            torch.save(unet.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.6f}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                # Load the best model before breaking
                unet.load_state_dict(torch.load(best_model_path))
                break

        # Plot and save the current loss curve
        plt.clf()
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{save_dir}/diffusion_loss_progress.png")

        # Visualize samples and denoising process periodically
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            generate_samples_grid(autoencoder, diffusion, epoch + 1, save_dir=save_dir)

        if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
            # Visualize denoising process for each class
            for target_class in range(10):  # Visualize paths to all classes
                visualize_denoising_steps(autoencoder, diffusion, epoch + 1,
                                          class_idx=target_class, save_dir=save_dir)

            # Save checkpoint
            torch.save(unet.state_dict(), f"{save_dir}/diffusion_model_epoch_{epoch + 1}.pt")

    plt.close()
    return unet, diffusion, loss_history


# Main function to run the entire pipeline
def main():
    # Set device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "./fashion_mnist_results"
    os.makedirs(results_dir, exist_ok=True)

    # Path for saved autoencoder
    autoencoder_path = f"{results_dir}/fashion_mnist_autoencoder.pt"

    # Create autoencoder
    autoencoder = SimpleAutoencoder(latent_dim=64, in_channels=1).to(device)

    # Check if trained autoencoder exists
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        autoencoder.eval()

        # Visualize reconstructions with loaded model
        visualize_reconstructions(autoencoder, epoch="loaded", save_dir=results_dir)
        visualize_latent_space(autoencoder, epoch="loaded", save_dir=results_dir)
    else:
        print("No existing autoencoder found. Training a new one...")
        autoencoder, ae_losses = train_autoencoder(
            autoencoder, num_epochs=20, lr=1e-4,
            visualize_every=1,  # Visualize every epoch
            save_dir=results_dir
        )

        # Save autoencoder
        torch.save(autoencoder.state_dict(), autoencoder_path)

        # Plot autoencoder loss
        plt.figure(figsize=(8, 5))
        plt.plot(ae_losses)
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/autoencoder_loss.png")
        plt.close()

    # Path for saved diffusion model
    diffusion_path = f"{results_dir}/fashion_mnist_diffusion.pt"

    # Create UNet
    unet = SimpleUNet(in_channels=1, hidden_dims=[32, 64, 128]).to(device)

    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    unet.apply(init_weights)

    # Check if trained diffusion model exists
    if os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = SimpleDenoiseDiffusion(unet, n_steps=1000, device=device)

        # Visualize samples with loaded model
        generate_samples_grid(autoencoder, diffusion, epoch="loaded", save_dir=results_dir)
        visualize_denoising_steps(autoencoder, diffusion, epoch="loaded", save_dir=results_dir)
    else:
        print("No existing diffusion model found. Training a new one...")
        unet, diffusion, diff_losses = train_diffusion(
            autoencoder, unet, num_epochs=100, lr=1e-3,
            visualize_every=100,  # Visualize every 10 epoch
            save_dir=results_dir
        )

        # Save diffusion model
        torch.save(unet.state_dict(), diffusion_path)

        # Plot diffusion loss
        plt.figure(figsize=(8, 5))
        plt.plot(diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()

    # Generate final visualization samples
    print("Generating final visualizations...")
    if 'diffusion' not in locals():
        # Create diffusion model if not already created
        diffusion = SimpleDenoiseDiffusion(unet, n_steps=1000, device=device)

    visualize_denoising_steps(autoencoder, diffusion, epoch="final", save_dir=results_dir)
    generate_samples_grid(autoencoder, diffusion, epoch="final", n_per_class=10, save_dir=results_dir)

    print(f"All processing complete. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
