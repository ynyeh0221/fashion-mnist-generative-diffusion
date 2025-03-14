import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import math

# Adjust input resolution for better balance between speed and detail
img_size = 16

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Downsample images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Optimize batch size and load Fashion MNIST instead of MNIST
batch_size = 128  # Adjusted for better training stability
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Define Fashion MNIST class names for better visualization
fashion_mnist_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


# Implement efficient attention mechanism (Linear Attention)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Get queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        # Apply scaling to queries
        q = q * self.scale

        # Apply softmax to keys (feature dimension)
        k = k.softmax(dim=-1)

        # Linear attention computation
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.dropout(self.to_out(out))


# Decrease model complexity with smaller transformer block
class EfficientTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EfficientTransformerBlock, self).__init__()
        # Use LinearAttention instead of standard MultiheadAttention
        self.attention = LinearAttention(embed_dim, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)

        # Feed forward with residual connection and layer normalization
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output
        return x


# Implement patch-based approach similar to ViT
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        x = self.proj(x)  # [batch_size, embed_dim, grid_size, grid_size]
        x = x.flatten(2)  # [batch_size, embed_dim, grid_size*grid_size]
        x = x.transpose(1, 2)  # [batch_size, grid_size*grid_size, embed_dim]
        return x


# Transformer-based Conditional Denoiser with optimizations
class EfficientTransformerDenoiser(nn.Module):
    def __init__(self, img_size=16, patch_size=2, num_classes=10, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256):
        super(EfficientTransformerDenoiser, self).__init__()

        # Save image and patch size as attributes
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch embedding instead of pixel-level tokens
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels=1, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Embeddings
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.noise_embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Reduced complexity - fewer layers, smaller embedding dim
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

        # Output head - simpler decoder
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x, noise_level, labels):
        batch_size = x.shape[0]

        # Get patch embeddings
        x = self.patch_embed(x)

        # Expand class token and concatenate
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Get label embeddings
        label_embed = self.label_embedding(labels).unsqueeze(1)

        # Process noise level
        noise_level_flat = noise_level.flatten().unsqueeze(1).expand(batch_size, 1)
        noise_embed = self.noise_embedding(noise_level_flat).unsqueeze(1)

        # Add conditions to class token
        x[:, 0] = x[:, 0] + label_embed.squeeze(1) + noise_embed.squeeze(1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Take the patch tokens only (exclude class token)
        x = x[:, 1:]
        x = self.norm(x)

        # Project each patch token to patch_size*patch_size pixels
        patch_pixels = self.projection(x)  # [batch_size, num_patches, patch_size*patch_size]

        # Reshape into image
        num_patches_per_side = int(self.img_size // self.patch_size)
        output = patch_pixels.view(batch_size, num_patches_per_side, num_patches_per_side, self.patch_size,
                                   self.patch_size)
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(batch_size, 1, self.img_size, self.img_size)

        return output


if __name__ == '__main__':
    # Initialize model, optimizer, loss function, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Current using device: {device}")

    # Decreased model complexity
    model = EfficientTransformerDenoiser(img_size=img_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)  # Adjusted for better convergence
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=1,
        eta_min=1e-6
    )

    # Training loop with multi-step diffusion
    epochs = 50
    num_diffusion_steps = 10

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Randomly select diffusion step
            t = random.randint(1, num_diffusion_steps)
            noise_level = torch.tensor([t / num_diffusion_steps], device=device).view(-1, 1, 1, 1)

            # Add noise based on noise level
            noisy_images = images + torch.randn_like(images).to(device) * noise_level

            optimizer.zero_grad()
            outputs = model(noisy_images, noise_level, labels)
            labels = labels[:, None, None, None].expand_as(outputs)  # Expand labels to match outputs
            # Current approach with scaling adjustment
            # Define these at the beginning of your script, before the training loop
            mse_scale = None
            l1_scale = None

            # Then inside your training loop, modify your loss calculation:
            # Calculate losses
            mse = mse_loss(outputs, images)
            l1 = l1_loss(outputs, images)

            # Initialize or update the scaling factors
            if mse_scale is None:
                # First batch
                mse_scale = mse.item()
                l1_scale = l1.item()
            else:
                # Smooth update (use a small factor to avoid rapid changes)
                mse_scale = 0.99 * mse_scale + 0.01 * mse.item()
                l1_scale = 0.99 * l1_scale + 0.01 * l1.item()

            # Scale factor to make L1 comparable to MSE
            scale_factor = mse_scale / l1_scale

            # Dynamic weighting with scaling
            mse_weight = 0.9 - 0.4 * (epoch / epochs)  # Gradually decrease MSE weight
            loss = (mse_weight * mse) + ((1 - mse_weight) * l1 * scale_factor)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # Update scheduler once per epoch, not per step

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # Step the scheduler once per epoch
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Evaluate and visualize multi-step generation for all fashion classes
    model.eval()

    with torch.no_grad():
        # Create a figure with 10 rows (one for each fashion class)
        fig, axes = plt.subplots(10, num_diffusion_steps + 1, figsize=(15, 25))

        # Generate each fashion item from 0-9
        for class_idx in range(10):
            # Start with random noise
            denoised_img = torch.randn((1, 1, img_size, img_size)).to(device)
            label = torch.tensor([class_idx], device=device)

            # Plot the starting noise
            axes[class_idx, 0].imshow(denoised_img.cpu().squeeze(), cmap='gray')
            axes[class_idx, 0].set_title(f"Start (Noise)")
            axes[class_idx, 0].axis('off')

            # Perform denoising steps
            for t in reversed(range(1, num_diffusion_steps + 1)):
                noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
                denoised_img = model(denoised_img, noise_level, label)

                # Resize output for visualization
                img_to_show = torch.nn.functional.interpolate(
                    denoised_img, size=(28, 28), mode='bilinear', align_corners=False
                ) if img_size != 28 else denoised_img

                # Plot the denoised image at this step
                axes[class_idx, num_diffusion_steps - t + 1].imshow(img_to_show.cpu().squeeze(), cmap='gray')
                axes[class_idx, num_diffusion_steps - t + 1].set_title(f"Step {num_diffusion_steps - t + 1}")
                axes[class_idx, 0].set_ylabel(f"{fashion_mnist_classes[class_idx]}", size='medium', rotation=0,
                                              labelpad=60, va='center', ha='right')
                axes[class_idx, num_diffusion_steps - t + 1].axis('off')

        plt.suptitle("Generation Process for Fashion MNIST Classes", fontsize=16)
        plt.subplots_adjust(left=0.2, wspace=0.3, hspace=0.3)
        plt.tight_layout()
        plt.savefig('fashion_mnist_generation.png', dpi=300, bbox_inches='tight')
        plt.show()
