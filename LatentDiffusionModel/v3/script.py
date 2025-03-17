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
from sklearn.decomposition import PCA

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set image size for Fashion MNIST (28x28 grayscale images)
img_size = 28

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # This already scales to [0,1] range
])

# Batch size for training
batch_size = 256

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


# Swish activation function
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


# Class embedding for conditional diffusion
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=10, n_channels=16):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, n_channels)
        self.lin1 = nn.Linear(n_channels, n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, c):
        # Get class embeddings
        emb = self.embedding(c)
        # Process through MLP (same structure as time embedding)
        return self.lin2(self.act(self.lin1(emb)))


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


# Residual block for UNet with class conditioning
class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=16, num_groups=8, dropout_rate=0.2):
        super().__init__()

        # Feature normalization and convolution
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time and class embedding projections
        self.time_emb = nn.Linear(d_time, out_channels)
        self.class_emb = nn.Linear(d_time, out_channels)  # Same dimension as time embedding
        self.act = Swish()

        self.dropout = nn.Dropout(dropout_rate)

        # Second convolution
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, c=None):
        # First part
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        t_emb = self.act(self.time_emb(t))
        h = h + t_emb.view(-1, t_emb.shape[1], 1, 1)

        # Add class embedding if provided
        if c is not None:
            c_emb = self.act(self.class_emb(c))
            h = h + c_emb.view(-1, c_emb.shape[1], 1, 1)

        # Second part
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual(x)


# Switch Sequential for handling time and class embeddings
class SwitchSequential(nn.Sequential):
    def forward(self, x, t=None, c=None):
        for layer in self:
            if isinstance(layer, UNetResidualBlock):
                x = layer(x, t, c)
            elif isinstance(layer, UNetAttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x


# Class-Conditional UNet for noise prediction
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[16, 32, 64], num_classes=10, dropout_rate=0.2):
        super().__init__()

        # Time embedding
        self.time_emb = TimeEmbedding(n_channels=16)

        # Class embedding
        self.class_emb = ClassEmbedding(num_classes=num_classes, n_channels=16)

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

        # Upsampling blocks
        for dim in reversed(hidden_dims[:-1]):
            self.up_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 4, stride=2, padding=1),
                    UNetResidualBlock(hidden_dims[-1] + dim, dim),
                    UNetResidualBlock(dim, dim),
                ])
            )
            hidden_dims[-1] = dim

        self.dropout_final = nn.Dropout(dropout_rate)

        # Final blocks
        self.final_block = SwitchSequential(
            UNetResidualBlock(hidden_dims[0] * 2, hidden_dims[0]),
            nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t, c=None):
        # Time embedding
        t_emb = self.time_emb(t)

        # Class embedding (if provided)
        c_emb = None
        if c is not None:
            c_emb = self.class_emb(c)

        # Initial convolution
        x = self.initial_conv(x)

        # Store skip connections
        skip_connections = [x]

        # Downsampling
        for resblock1, resblock2, downsample in self.down_blocks:
            x = resblock1(x, t_emb, c_emb)
            x = resblock2(x, t_emb, c_emb)
            skip_connections.append(x)
            x = downsample(x)

        x = self.dropout_mid(x)

        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, UNetAttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb, c_emb)

        # Upsampling
        for upsample, resblock1, resblock2 in self.up_blocks:
            x = upsample(x)  # Upsample first
            x = torch.cat([x, skip_connections.pop()], dim=1)  # Then concatenate
            x = resblock1(x, t_emb, c_emb)  # Then process
            x = resblock2(x, t_emb, c_emb)

        x = self.dropout_final(x)

        # Final blocks
        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_block(x, t_emb, c_emb)

        return x


# Class-conditional diffusion model
# Class-conditional diffusion model
class ConditionalDenoiseDiffusion():
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

    def p_sample(self, xt, t, c=None):
        """Single denoising step with optional class conditioning"""
        # Convert time to tensor format expected by model
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)

        # Predict noise (with class conditioning if provided)
        eps_theta = self.eps_model(xt, t, c)

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

    def sample(self, shape, device, c=None):
        """Generate samples by denoising from pure noise with optional class conditioning"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Progressively denoise with class conditioning
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, c)

        return x

    def loss(self, x0, labels=None):
        """Calculate noise prediction loss with optional class conditioning"""
        batch_size = x0.shape[0]

        # Random timestep for each sample
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Add noise
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)

        # Predict noise (with class conditioning if labels provided)
        eps_theta = self.eps_model(xt, t, labels)

        return euclidean_distance_loss(eps, eps_theta)

# Function to generate a grid of samples for all classes
def generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir="./results"):
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
    fig.suptitle(f'Fashion MNIST Samples Generated by Diffusion Model',
                 fontsize=16, y=0.98)

    for i in range(n_classes):
        # Create a text-only cell for the class name
        axes[i, 0].text(0.5, 0.5, class_names[i],
                        fontsize=14, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        axes[i, 0].axis('off')

        # Generate samples with class conditioning
        class_tensor = torch.tensor([i] * n_per_class, device=device)
        latent_shape = (n_per_class, 1, 8, 8)

        # Sample from the diffusion model with class conditioning
        samples = diffusion.sample(latent_shape, device, class_tensor)

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
        "This visualization shows fashion items generated by the conditional diffusion model.\n"
        "The model creates new, synthetic images based on learned patterns from Fashion MNIST.\n"
        "Each row corresponds to a different clothing category as labeled."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to make room for titles
    plt.savefig(f"{save_dir}/samples_grid_all_classes.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    print(f"Generated sample grid for all classes with clearly labeled fashion categories")
    return f"{save_dir}/samples_grid_all_classes.png"


# Visualize latent space denoising process for a specific class
def visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=None):
    """
    Visualize both the denoising process and the corresponding path in latent space.

    Args:
        autoencoder: Trained autoencoder model
        diffusion: Trained diffusion model
        class_idx: Target class index (0-9)
        save_dir: Directory to save visualizations
    """
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    # ===== PART 1: Setup dimensionality reduction for latent space =====
    print(f"Generating latent space projection for class {class_names[class_idx]}...")
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
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(all_latents)

    # ===== PART 2: Setup denoising visualization =====
    # Parameters for visualization
    n_samples = 5  # Number of samples to generate
    steps_to_show = 8  # Number of denoising steps to visualize
    step_size = diffusion.n_steps // steps_to_show
    timesteps = list(range(0, diffusion.n_steps, step_size))[::-1]

    # Generate sample from pure noise with class conditioning
    class_tensor = torch.tensor([class_idx] * n_samples, device=device)
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

            # Denoise from current step to t=0 with class conditioning
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device), class_tensor)

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

    # Set title for the denoising subplot
    ax_denoising.set_title(f"Diffusion Model Denoising Process for {class_names[class_idx]}",
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

    # Highlight target class area
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

    ax_latent.set_title(f"Diffusion Path in Latent Space for {class_names[class_idx]}", fontsize=16)
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
    safe_class_name = class_names[class_idx].replace('/', '-')

    # Save the figure
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Denoising visualization for {class_names[class_idx]} saved to {save_path}")

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    return save_path


# Visualization functions that were missing

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
    except Exception as e:
        print(f"t-SNE visualization error: {e}")

    autoencoder.train()


# Function to generate samples of a specific class (need this for training)
def generate_class_samples(autoencoder, diffusion, target_class, num_samples=5, save_path=None):
    """
    Generate samples of a specific target class

    Args:
        autoencoder: Trained autoencoder model
        diffusion: Trained conditional diffusion model
        target_class: Index of the target class (0-9) or class name
        num_samples: Number of samples to generate
        save_path: Path to save the generated samples

    Returns:
        Tensor of generated samples
    """
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    # Convert class name to index if string is provided
    if isinstance(target_class, str):
        if target_class in class_names:
            target_class = class_names.index(target_class)
        else:
            raise ValueError(f"Invalid class name: {target_class}. Must be one of {class_names}")

    # Create class conditioning tensor
    class_tensor = torch.tensor([target_class] * num_samples, device=device)

    # Generate samples
    latent_shape = (num_samples, 1, 8, 8)
    with torch.no_grad():
        # Sample from the diffusion model with class conditioning
        latent_samples = diffusion.sample(latent_shape, device, class_tensor)

        # Decode latents to images
        latent_samples_flat = latent_samples.view(num_samples, -1)
        samples = autoencoder.decode(latent_samples_flat)

    # Save samples if path provided
    if save_path:
        plt.figure(figsize=(num_samples * 2, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(samples[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"{class_names[target_class]}")

        plt.suptitle(f"Generated {class_names[target_class]} Samples")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return samples

# Main function
def main():
    """Main function to run the entire pipeline non-interactively"""
    print("Starting class-conditional diffusion model for Fashion MNIST")

    # Set device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "./fashion_mnist_conditional"
    os.makedirs(results_dir, exist_ok=True)

    # Load Fashion MNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Paths for saved models
    autoencoder_path = f"{results_dir}/fashion_mnist_autoencoder.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"

    # Create autoencoder
    autoencoder = SimpleAutoencoder(latent_dim=64, in_channels=1).to(device)

    # Check if trained autoencoder exists
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        autoencoder.eval()
    else:
        print("No existing autoencoder found. Training a new one...")

        # Define train function
        def train_autoencoder(autoencoder, num_epochs=20, lr=1e-4, visualize_every=1, save_dir=results_dir):
            print("Starting Autoencoder training...")
            os.makedirs(save_dir, exist_ok=True)

            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            loss_history = []

            for epoch in range(num_epochs):
                epoch_loss = 0
                autoencoder.train()

                for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    data = data.to(device)

                    # Forward pass
                    reconstructed = autoencoder(data)
                    loss = euclidean_distance_loss(reconstructed, data)

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
                    visualize_reconstructions(autoencoder, epoch + 1, save_dir)
                    visualize_latent_space(autoencoder, epoch + 1, save_dir)

                    # Save checkpoint
                    torch.save(autoencoder.state_dict(), f"{save_dir}/autoencoder_epoch_{epoch + 1}.pt")

            return autoencoder, loss_history

        # Train autoencoder
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

    # Create conditional UNet
    conditional_unet = ConditionalUNet(
        in_channels=1,
        hidden_dims=[32, 64, 128],
        num_classes=10
    ).to(device)

    # Initialize weights for UNet if needed
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Check if trained diffusion model exists
    if os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        conditional_unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
    else:
        print("No existing diffusion model found. Training a new one...")
        conditional_unet.apply(init_weights)

        # Define train function
        def train_conditional_diffusion(autoencoder, unet, num_epochs=100, lr=1e-3, visualize_every=10, save_dir=results_dir):
            print("Starting Class-Conditional Diffusion Model training...")
            os.makedirs(save_dir, exist_ok=True)

            autoencoder.eval()  # Set autoencoder to evaluation mode

            # Create diffusion model
            diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)
            optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # Training loop
            loss_history = []

            for epoch in range(num_epochs):
                epoch_loss = 0

                for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    data = data.to(device)
                    labels = labels.to(device)

                    # Encode images to latent space
                    with torch.no_grad():
                        latents = autoencoder.encode(data)
                        # Reshape latents to spatial form [B, 1, 8, 8]
                        latents = latents.view(-1, 1, 8, 8)

                    # Calculate diffusion loss with class conditioning
                    loss = diffusion.loss(latents, labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                # Calculate average loss
                avg_loss = epoch_loss / len(train_loader)
                loss_history.append(avg_loss)
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

                # Learning rate scheduling
                scheduler.step(avg_loss)

                # Visualize samples periodically
                if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                    # Generate samples for a couple of classes
                    for class_idx in [0, 5]:  # Just visualize two classes during training
                        save_path = f"{save_dir}/class_{class_names[class_idx].replace('/', '-')}_epoch_{epoch + 1}.png"
                        generate_class_samples(autoencoder, diffusion, target_class=class_idx, num_samples=5, save_path=save_path)
                        save_path = f"{save_dir}/denoising_path_{class_names[class_idx].replace('/', '-')}_epoch_{epoch}.png"
                        visualize_denoising_steps(autoencoder, diffusion, class_idx=class_idx, save_path=save_path)

                    # Save checkpoint
                    torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")

            return unet, diffusion, loss_history

        # Train conditional diffusion model
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, num_epochs=100, lr=1e-3,
            visualize_every=1,  # Visualize every 10 epochs
            save_dir=results_dir
        )

        # Save diffusion model
        torch.save(conditional_unet.state_dict(), diffusion_path)

        # Plot diffusion loss
        plt.figure(figsize=(8, 5))
        plt.plot(diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()

    # Make sure diffusion is defined
    if 'diffusion' not in locals():
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)

    # Generate sample grid for all classes
    print("Generating sample grid for all classes...")
    grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir=results_dir)
    print(f"Sample grid saved to: {grid_path}")

    # Generate denoising visualizations for all classes
    print("Generating denoising visualizations for all classes...")
    denoising_paths = []
    for class_idx in range(len(class_names)):
        path = visualize_denoising_steps(autoencoder, diffusion, class_idx, save_dir=results_dir)
        denoising_paths.append(path)
        print(f"Generated visualization for {class_names[class_idx]}")

    print("\nAll visualizations complete!")
    print(f"Sample grid: {grid_path}")
    print("Denoising visualizations:")
    for i, path in enumerate(denoising_paths):
        print(f"  - {class_names[i]}: {path}")


if __name__ == "__main__":
    main()
