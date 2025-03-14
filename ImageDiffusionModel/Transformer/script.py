import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset and split into train/validation sets
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Increase batch size for better training efficiency - use larger batch size if memory allows
batch_size = 256  # Increased from 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2 if torch.cuda.is_available() else 0,
                          pin_memory=True)  # Added pin_memory for faster data transfer
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         pin_memory=True)


# Optimized Conditional Transformer Denoiser with reduced size
class OptimizedDenoiser(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        super(OptimizedDenoiser, self).__init__()

        # Simplified label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Simplified noise level embedding
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Efficient feature extractor - UNet-style downsampling
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),  # SiLU (Swish) can be faster and more effective than GELU
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, embed_dim, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.GroupNorm(32, embed_dim),
            nn.SiLU(),
        )

        # Fixed position encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))

        # Optimized Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simplified decoder - upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, noise_level, labels):
        B, C, H, W = x.shape

        # Extract features
        features = self.feature_extractor(x)  # [B, embed_dim, 7, 7]

        # Reshape to sequence
        features = features.flatten(2).permute(0, 2, 1)  # [B, 49, embed_dim]

        # Add positional embeddings
        features = features + self.positional_embedding

        # Noise level embedding - simplified
        noise_embed = self.noise_embedding(noise_level.view(B, 1))  # [B, embed_dim]
        noise_embed = noise_embed.unsqueeze(1).expand(-1, 49, -1)  # [B, 49, embed_dim]

        # Label embedding - simplified
        label_embed = self.label_embedding(labels)  # [B, embed_dim]
        label_embed = self.label_proj(label_embed)  # [B, embed_dim]
        label_embed = label_embed.unsqueeze(1).expand(-1, 49, -1)  # [B, 49, embed_dim]

        # Combine all inputs
        transformer_input = features + noise_embed + label_embed

        # Apply Transformer
        transformer_output = self.transformer_encoder(transformer_input)

        # Reshape back to spatial dimensions
        spatial_features = transformer_output.permute(0, 2, 1).reshape(B, -1, 7, 7)

        # Decode to original image size
        output = self.decoder(spatial_features)

        return output


# Faster visualization function - only show a subset of classes and steps
def visualize_samples(model, device, epoch, save_path=None):
    model.eval()
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Show fewer classes and steps for faster visualization
    num_classes_to_show = 3  # Reduced from 5
    num_diffusion_steps = 5  # Reduced from 10

    fig, axes = plt.subplots(num_classes_to_show, num_diffusion_steps + 1, figsize=(12, 6))

    with torch.no_grad():
        for i in range(num_classes_to_show):
            prompt_label = i
            label = torch.tensor([prompt_label], device=device)

            # Start with random noise
            denoised_img = torch.randn((1, 1, 28, 28)).to(device)

            axes[i, 0].imshow(denoised_img.cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title("Initial Noise")
            axes[i, 0].axis('off')

            # Sequential denoising
            for t in reversed(range(1, num_diffusion_steps + 1)):
                noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
                denoised_img = model(denoised_img, noise_level, label)

                axes[i, num_diffusion_steps - t + 1].imshow(denoised_img.cpu().squeeze(), cmap='gray')
                if i == 0:
                    axes[i, num_diffusion_steps - t + 1].set_title(f"Step {num_diffusion_steps - t + 1}")

                if num_diffusion_steps - t + 1 == num_diffusion_steps:
                    axes[i, num_diffusion_steps].set_ylabel(class_labels[prompt_label], rotation=0, labelpad=40,
                                                            fontsize=10)

                axes[i, num_diffusion_steps - t + 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/samples_epoch_{epoch}.png")
    plt.show()


# Simplified evaluation function
def evaluate(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0
    batches = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            # Limit evaluation batches for speed if specified
            if max_batches is not None and i >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Fixed mid-level noise for consistent evaluation
            noise_level = torch.ones(images.size(0), 1, 1, 1).to(device) * 0.3
            noisy_images = images + torch.randn_like(images).to(device) * noise_level

            outputs = model(noisy_images, noise_level, labels)

            # Simpler loss function for faster evaluation
            loss = F.mse_loss(outputs, images)

            total_loss += loss.item()
            batches += 1

    return total_loss / batches


# Efficient combined loss function
def fast_combined_loss(outputs, targets, mse_weight=0.8, l1_weight=0.2):
    mse = F.mse_loss(outputs, targets)
    l1 = F.l1_loss(outputs, targets)
    # Removed cosine similarity loss for speed
    return mse_weight * mse + l1_weight * l1


# Check if mixed precision is available
try:
    from torch.cuda.amp import autocast, GradScaler

    mixed_precision_available = torch.cuda.is_available()
except ImportError:
    mixed_precision_available = False
    print("Mixed precision training not available")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else
                      'cpu')
print(f"Using device: {device}")

# Create model instance with smaller size
model = OptimizedDenoiser().to(device)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Improved optimization settings
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Higher initial learning rate

# Reduced training epochs
epochs = 15  # Reduced from 30
warmup_epochs = 2  # Reduced from 5

# More aggressive learning rate schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,  # Higher max learning rate
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,  # Faster warmup
    div_factor=25.0,
    final_div_factor=1000.0
)

# Gradient clipping
max_grad_norm = 1.0

# Initialize grad scaler for mixed precision if available
scaler = GradScaler() if mixed_precision_available else None

# Training and validation history
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 5  # For early stopping
early_stop_counter = 0

# Create directory for saving
save_dir = './fashion_denoiser_optimized'
os.makedirs(save_dir, exist_ok=True)

# Improved training loop
print("Starting training...")
training_start_time = time.time()

try:
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Progressive noise strategy
            if epoch < warmup_epochs:
                # Start with small noise in early epochs
                noise_level = (0.2 * torch.rand(images.size(0), 1, 1, 1) + 0.05).to(device)
            else:
                # Gradually increase noise level
                noise_level = (0.8 * torch.rand(images.size(0), 1, 1, 1) + 0.1).to(device)

            noise = torch.randn_like(images).to(device) * noise_level
            noisy_images = images + noise

            optimizer.zero_grad()

            # Use mixed precision if available
            if mixed_precision_available:
                with autocast():
                    outputs = model(noisy_images, noise_level, labels)
                    loss = fast_combined_loss(outputs, images)

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(noisy_images, noise_level, labels)
                loss = fast_combined_loss(outputs, images)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            # Print progress less frequently
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Less frequent validation (every other epoch)
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            # Use partial validation for speed (25% of validation set)
            max_val_batches = max(1, len(val_loader) // 4)
            val_loss = evaluate(model, val_loader, device, max_batches=max_val_batches)
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"Validation loss did not improve. Early stopping counter: {early_stop_counter}/{patience}")

            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        else:
            # If not validating this epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")

        # Periodically visualize results (less frequently)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            visualize_samples(model, device, epoch + 1, save_path=save_dir)

        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    # Total training time
    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

except KeyboardInterrupt:
    print("Training interrupted by user")
    # Save checkpoint on interrupt
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, f"{save_dir}/interrupted_checkpoint.pt")

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
if val_losses:  # Only plot if we have validation losses
    # Create proper x-axis for validation losses (they might be fewer due to less frequent validation)
    # FIX: Generate val_epochs correctly based on when validation was performed
    val_epochs = [e for e in range(epochs) if e == 0 or (e + 1) % 2 == 0 or e == epochs - 1]
    val_epochs = val_epochs[:len(val_losses)]  # In case training was interrupted
    plt.plot(val_epochs, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f"{save_dir}/loss_curves.png")
plt.show()

# Load best model for final evaluation and visualization
try:
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))
    print("Loaded best model for evaluation")
except:
    print("Could not load best model, using current model state")

# Final evaluation
test_loss = evaluate(model, test_loader, device)
print(f"Final test loss: {test_loss:.4f}")

# Generate final samples
print("Generating final samples...")
visualize_samples(model, device, epochs, save_path=save_dir)


# Visualization of different diffusion steps
def visualize_diffusion_steps_comparison(model, device):
    model.eval()
    step_options = [5, 10, 20]  # Reduced from [5, 10, 20, 50]
    num_classes = 2

    fig, axes = plt.subplots(len(step_options), num_classes, figsize=(8, 9))
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for i, steps in enumerate(step_options):
            for j in range(num_classes):
                # Use same random seed for fair comparison
                torch.manual_seed(42 + j)
                denoised_img = torch.randn((1, 1, 28, 28)).to(device)

                label = torch.tensor([j], device=device)

                # Sequential denoising
                for t in reversed(range(1, steps + 1)):
                    noise_level = torch.tensor([[[[t / steps]]]], device=device)
                    denoised_img = model(denoised_img, noise_level, label)

                axes[i, j].imshow(denoised_img.cpu().squeeze(), cmap='gray')

                if j == 0:
                    axes[i, j].set_ylabel(f"{steps} steps", rotation=90, labelpad=15)

                if i == 0:
                    axes[i, j].set_title(class_labels[j])

                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/diffusion_steps_comparison.png")
    plt.show()


# Compare different numbers of diffusion steps
visualize_diffusion_steps_comparison(model, device)

print("Training and evaluation complete!")
