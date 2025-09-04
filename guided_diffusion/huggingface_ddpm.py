
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np


image_size = 32
in_channels = 1
out_channels = 1
epochs = 50
batch_size = 128
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
os.makedirs("./mnist_checkpoints", exist_ok=True)
os.makedirs("./training_samples", exist_ok=True)

model = UNet2DModel(
    sample_size=image_size,
    in_channels=in_channels,
    out_channels=out_channels,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=(
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",
        "AttnDownBlock2D"
    ),
    up_block_types=(
        "AttnUpBlock2D", 
        "AttnUpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"
    ),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_history = []
epoch_history = []
def uncondit_samples(model, scheduler, num_samples=4, steps=100):
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(num_samples, 1, image_size, image_size, device=device)
        
        for t in reversed(range(steps)):
            t_tensor = torch.tensor([t] * num_samples, device=device, dtype=torch.long)
            noise_pred = model(x, t_tensor).sample
            
            alpha_t = scheduler.alphas_cumprod[t].to(device)
            beta_t = 1 - alpha_t
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise

        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
    
    model.train()
    return x.cpu()

for epoch in range(epochs):
    epoch_losses = []
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for step, batch in enumerate(pbar):
        clean_images = batch[0].to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, 
                                (clean_images.shape[0],), device=device).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        current_loss = np.mean(epoch_losses[-100:])  # Moving average
        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

    avg_epoch_loss = np.mean(epoch_losses)
    loss_history.extend(epoch_losses)
    epoch_history.extend([epoch + 1] * len(epoch_losses))
    
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        # Save model
        model.save_pretrained(f"./mnist_checkpoints/epoch_{epoch+1}")
        noise_scheduler.save_pretrained(f"./mnist_checkpoints/epoch_{epoch+1}")
   
        samples = uncondit_samples(model, noise_scheduler, num_samples=8)
        
        plt.figure(figsize=(12, 3))
        for i in range(8):
            plt.subplot(1, 8, i+1)
            plt.imshow(samples[i, 0], cmap="gray")
            plt.axis('off')
            plt.title(f"Epoch {epoch+1}")
        
        plt.tight_layout()
        plt.savefig(f"./training_samples/epoch_{epoch+1}.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved checkpoint and samples for epoch {epoch+1}")

model.save_pretrained("./mnist_unet")
noise_scheduler.save_pretrained("./mnist_scheduler")

plt.figure(figsize=(10, 6))
plt.plot(loss_history, alpha=0.6, label="Batch Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Progression")
plt.legend()
plt.grid(True)
plt.savefig("./training_loss.png", dpi=100, bbox_inches='tight')
plt.show()

