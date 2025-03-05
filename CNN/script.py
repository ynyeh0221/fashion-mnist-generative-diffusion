import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# Data loading and preprocessing for Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# Corrected Conditional CNN Denoiser Model
class ConditionalDenoiser(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalDenoiser, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 64)
        self.label_fc = nn.Linear(64, 28 * 28)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, noise_level, labels):
        label_embedding = self.label_fc(self.label_embedding(labels)).view(-1, 1, 28, 28)
        noise_channel = noise_level.expand_as(x)
        input_combined = torch.cat([x, noise_channel, label_embedding], dim=1)
        encoded = self.encoder(input_combined)
        decoded = self.decoder(encoded)
        return decoded


# Device setting
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = ConditionalDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.MSELoss()

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        noise_level = (0.4 * torch.rand(images.size(0), 1, 1, 1) + 0.1).to(device)
        noisy_images = images + torch.randn_like(images).to(device) * noise_level

        optimizer.zero_grad()
        outputs = model(noisy_images, noise_level, labels)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Inference and visualization
model.eval()
prompt_label = random.randint(0, 9)
num_diffusion_steps = 10

with torch.no_grad():
    denoised_img = torch.randn((1, 1, 28, 28)).to(device)
    label = torch.tensor([prompt_label], device=device)

    fig, axes = plt.subplots(1, num_diffusion_steps + 1, figsize=(15, 3))
    axes[0].imshow(denoised_img.cpu().squeeze(), cmap='gray')
    axes[0].set_title("Initial Noise")
    axes[0].axis('off')

    for t in reversed(range(1, num_diffusion_steps + 1)):
        noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
        label_embedding = model.label_fc(model.label_embedding(label)).view(-1, 1, 28, 28)
        noise_channel = noise_level.expand_as(denoised_img)
        input_combined = torch.cat([denoised_img, noise_channel, label_embedding], dim=1)
        encoded_img = model.encoder(input_combined)
        denoised_img = model.decoder(encoded_img)

        axes[num_diffusion_steps - t + 1].imshow(denoised_img.cpu().squeeze(), cmap='gray')
        axes[num_diffusion_steps - t + 1].set_title(f"Step {num_diffusion_steps - t + 1}\nClass {prompt_label}")
        axes[num_diffusion_steps - t + 1].axis('off')

plt.tight_layout()
plt.show()
