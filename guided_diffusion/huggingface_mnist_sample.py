import sys
import os
import torch
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt

os.environ["HF_HUB_OFFLINE"] = "1"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




model_path = r"C:\Users\hesal5042\OneDrive - University of Bergen\Research\NORCE\hello\mnist_unet"
scheduler_path = r"C:\Users\hesal5042\OneDrive - University of Bergen\Research\NORCE\hello\mnist_scheduler"


model = UNet2DModel.from_pretrained(model_path).to(device)
model.eval()
scheduler = DDPMScheduler.from_pretrained(scheduler_path)
scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


B = 8
image = torch.stack([mnist[i][0] for i in range(B)]).to(device)  # shape of 8,1,32,32 = [B, 1, 32, 32]
labels = torch.tensor([mnist[i][1] for i in range(B)]).to(device)
print("Image batch shape:", image.shape)


mask = torch.ones_like(image)
mask[:, :, :, 16:] = 0

# RePaint here---------
def repaint(model, scheduler, x, mask, steps=100, jump_length=5, jump_n_sample=5):
    B = x.size(0)
    device = x.device
    x_t = torch.randn_like(x)  # start from pure noise

    for t_idx in reversed(range(steps)):
        t = t_idx  

        with torch.no_grad():
      
            model_output = model(x_t, torch.full((B,), t, device=device, dtype=torch.long)).sample

        step_output = scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=x_t
        )
        x_t = step_output.prev_sample

        
        x_t = x_t * (1 - mask) + x * mask

        # Forward jump
        if (t_idx % jump_length == 0) and (t_idx > 0):
            for _ in range(jump_n_sample):
                noise = torch.randn_like(x_t)
                t_tensor = torch.tensor([t], device=x_t.device)
                x_t = scheduler.add_noise(x_t, noise, t_tensor)
                #x_t = x_t * (1 - mask) + x * mask
                x_t = x_t * 1  + x #removed mask

    return x_t


output = repaint(model, scheduler, image, mask, steps=100, jump_length=10, jump_n_sample=10)

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image[0, 0].cpu(), cmap="gray")
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow((image * mask)[0, 0].cpu(), cmap="gray")
plt.title("Masked")

plt.subplot(1, 3, 3)
plt.imshow(output[0, 0].detach().cpu(), cmap="gray")
plt.title("RePainted")
plt.show()
