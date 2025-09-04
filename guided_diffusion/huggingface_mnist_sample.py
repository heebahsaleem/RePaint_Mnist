import sys
import os
import torch
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import random

os.environ["HF_HUB_OFFLINE"] = "1"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




model_path = r"mnist_unet"
scheduler_path = r"mnist_scheduler"


model = UNet2DModel.from_pretrained(model_path).to(device)
model.eval()
scheduler = DDPMScheduler.from_pretrained(scheduler_path)
alphas = scheduler.alphas_cumprod.to(device)#storing scheduler parameter


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# B = 8
B=4 #reduing batch size for better visualization
# image = torch.stack([mnist[i][0] for i in range(B)]).to(device)  # shape of 8,1,32,32 = [B, 1, 32, 32]
# labels = torch.tensor([mnist[i][1] for i in range(B)]).to(device)
#fixed issue for random images
indices = random.sample(range(len(mnist)), B)#random unique indices
image = torch.stack([mnist[i][0] for i in indices]).to(device)
labels = torch.tensor([mnist[i][1] for i in indices]).to(device)
print("Image batch shape:", image.shape)


mask = torch.ones_like(image)
mask[:, :, :, 16:] = 0
known_region = image * mask #adding this
# RePaint here---------
def repaint(model, scheduler, x, mask, steps=250, jump_length=15, jump_n_sample=8):
    """
    x_known: original image with masked regions set to 0
    mask: binary mask (1 = known, 0 = unknown)
    """
    device = x.device
    B = x.size(0)
    
    # Start from pure noise
    x_t = torch.randn_like(x)
    
   
    
    for t in reversed(range(steps)):

        t_tensor = torch.tensor([t] * B, device=device, dtype=torch.long)
        
  
        with torch.no_grad():#predictng noise
            noise_pred = model(x_t, t_tensor).sample
        

        alpha_t = alphas[t]
        beta_t = 1 - alpha_t
        

        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        #reverse formula
        x_t_prev = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred
        ) + torch.sqrt(beta_t) * noise

        x_t_prev = mask * x + (1 - mask) * x_t_prev

        if (t % jump_length == 0) and (t > 0): #forward jumpp
            for _ in range(jump_n_sample):

                noise_jump = torch.randn_like(x_t_prev)
                x_t_prev = torch.sqrt(alpha_t) * x_t_prev + torch.sqrt(1 - alpha_t) * noise_jump

                x_t_prev = mask * x + (1 - mask) * x_t_prev
        
        x_t = x_t_prev
    

    output = (x_t + 1) / 2
    output = torch.clamp(output, 0, 1)
    
    return output
        
        # step_output = scheduler.step(
        #     model_output=model_output,
        #     timestep=t,
        #     sample=x_t
        # )
        # x_t = step_output.prev_sample

        
        # x_t = x_t * (1 - mask) + x * mask
        # x_t = mask * x + (1 - mask) * x_t
        # x_t = x_t * 1 + x
        # x_t = x_t * (1 - mask) + x * mask
        # x_t+1 == x_t * mask + x * (1 - mask)
        # x_t = mask * x + (1 - mask) * x_t


        # Forward jump
    #     if (t_idx % jump_length == 0) and (t_idx > 0):
    #         for _ in range(jump_n_sample):
    #             noise = torch.randn_like(x_t)
    #             t_tensor = torch.tensor([t], device=x_t.device)
    #             x_t = scheduler.add_noise(x_t, noise, t_tensor)
    #             # x_t = x_t * (1 - mask) + x * mask
    #             # x_t = x_t * 1  + x #removed mask
    #             # x_t = x_t * 1 + x
    #             # x_t = scheduler.add_noise(x_t, noise, t_tensor)
    #             # x_t = x_t * mask + x * (1 - mask)
    #             x_t = mask * x + (1 - mask) * x_t
    # x_t = torch.clamp(x_t, 0, 1)
    # return x_t


#     output = repaint(model, scheduler, known_region, mask, params)

output = repaint(model, scheduler, known_region, mask, steps=300, jump_length=20, jump_n_sample=10)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image[0, 0].cpu(), cmap="gray",vmin=0, vmax=1)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow((image * mask)[0, 0].cpu(), cmap="gray",vmin=0, vmax=1)
plt.title("Masked")

plt.subplot(1, 3, 3)
plt.imshow(output[0, 0].detach().cpu(), cmap="gray",vmin=0, vmax=1)
# plt.imshow(normalize(output[0,0].detach().cpu()), cmap="gray")
plt.title("RePainted")
plt.show()

#as kristian suggested to also check without masking or unconditional generation to verify model quality
#after sampling, it will test unconditional generation 
print("this is unconditiional gneration testing")

with torch.no_grad():
    unconditional_noise = torch.randn(4, 1, 32, 32, device=device)
    unconditional_samples = []
    
    for t in reversed(range(100)):
        t_tensor = torch.tensor([t] * 4, device=device, dtype=torch.long)
        noise_pred = model(unconditional_noise, t_tensor).sample
        
        alpha_t = scheduler.alphas_cumprod[t].to(device)
        beta_t = 1 - alpha_t
        print("eorking")
        if t > 0:
            noise = torch.randn_like(unconditional_noise)
        else:
            noise = torch.zeros_like(unconditional_noise)
        
        unconditional_noise = (1 / torch.sqrt(alpha_t)) * (
            unconditional_noise - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred
        ) + torch.sqrt(beta_t) * noise
    
    unconditional_output = (unconditional_noise + 1) / 2
    unconditional_output = torch.clamp(unconditional_output, 0, 1)

plt.figure(figsize=(10, 2.5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(unconditional_output[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    plt.title("Unconditional")
    plt.axis('off')
plt.tight_layout()
plt.show()