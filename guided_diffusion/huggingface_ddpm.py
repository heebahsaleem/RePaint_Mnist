# import torch
# import torchvision
# from torch import nn
# from torchvision import datasets, transforms
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from diffusers import DDPMScheduler, UNet2DModel
# from matplotlib import pyplot as plt
# from diffusers import DDPMPipeline
# import torch.optim as optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# # pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
# # pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# # Load MNIST dataset
# mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
# mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# from diffusers import UNet2DModel

# model = UNet2DModel(
#     sample_size=32,       # Image size
#     in_channels=1,        # MNIST is grayscale
#     out_channels=1,       # Predicting noise for each pixel
#     layers_per_block=2,
#     block_out_channels=(64, 128, 128, 256),
#     down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
#     up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
# )

# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# from diffusers import DDPMScheduler

# scheduler = DDPMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.0001,
#     beta_end=0.02,
#     beta_schedule="linear"
# )

# for epoch in range(5):
#     for batch in mnist_loader:
#         images, _ = batch
#         images = images.to(device)
        
#         noise = torch.randn_like(images)
#         timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.size(0),), device=device)
        
#         noisy_images = scheduler.add_noise(images, noise, timesteps)
#         predicted_noise = model(noisy_images, timesteps).sample
        
#         loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
#     torch.save(model.state_dict(), 
#     f"C:/Users/hesal5042/OneDrive - University of Bergen/Research/NORCE/hello/RePaint/guided_diffusion_mnist/huggingface_checkpoints/mnist_ddpm_epoch{epoch+1}.pt")


from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

image_size = 32  # MNIST
in_channels = 1  # grayscale
out_channels = 1
base_channels = 64

model = UNet2DModel(
    sample_size=image_size,   # target image size
    in_channels=in_channels,  # MNIST is grayscale
    out_channels=out_channels,
    layers_per_block=2,
    block_out_channels=(base_channels, base_channels*2, base_channels*4),
    down_block_types=(
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D"
    ),
    up_block_types=(
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
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

loss_history = []
epoch_history = []

epochs = 5  # you can increase

for epoch in range(epochs):
    pbar = tqdm(train_dataloader)
    for step, batch in enumerate(pbar):
        clean_images = batch[0].to(device)  # images only, labels ignored
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=device).long()

        # add noise
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # predict noise
        noise_pred = model(noisy_images, timesteps).sample

        loss = F.mse_loss(noise_pred, noise)

        loss_history.append(loss.item())
        epoch_history.append(epoch + step / len(train_dataloader))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        
    model.save_pretrained("./mnist_unet")
    noise_scheduler.save_pretrained("./mnist_scheduler")


plt.figure(figsize=(8,5))
plt.plot(epoch_history, loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.show()


