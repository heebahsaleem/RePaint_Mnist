import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from guided_diffusion.script_util import create_model_and_diffusion
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Config (must match your training config!)
# model_config = {
#     "image_size":32,
#     "num_channels":64,
#     "num_res_blocks":2,
#     "channel_mult":"",
#     "learn_sigma":True,
#     "class_cond":False,
#     # num_classes=10,
#     "diffusion_steps":1000,
#     "noise_schedule":"linear",
#     "timestep_respacing":"",
#     "use_kl":False,
#     "predict_xstart": True,
#     "rescale_timesteps":True,
#     "rescale_learned_sigmas":True,
#     "use_checkpoint":False,
#     "use_scale_shift_norm":True,
#     "resblock_updown":False,
#     "use_fp16":False,
#     "num_heads":1,
#     "num_head_channels":64,
#     "num_heads_upsample":-1,
#     "use_new_attention_order":False,
#     "attention_resolutions":"16",
#     "dropout":0.1
    

# }

model_path = r"C:\Users\hesal5042\OneDrive - University of Bergen\Research\NORCE\hello\RePaint\guided_diffusion_mnist\mnist_ddpm_epoch10.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

CUDA_LAUNCH_BLOCKING=1
# Create model + diffusion
model, diffusion = create_model_and_diffusion(
    image_size=32,
    num_channels=64,
    num_res_blocks=2,
    channel_mult="", #will try to change to    channel_mult=[1, 2, 2], and for training also.
    learn_sigma=True,
    class_cond=False,
    # num_classes=10,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="100",
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=True,
    rescale_learned_sigmas=True,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_fp16=False,
    num_heads=1,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_new_attention_order=False,
    attention_resolutions="16",
    dropout=0.1
)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
# print(model)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
# image, label = mnist[0]  # get one image --> tried on only image 7
image = torch.stack([mnist[i][0] for i in range(8)]).to(device) # shape of 8,1,32,32. batch of images ready for model
labels =torch.tensor([mnist[i][1] for i in range(8)]) #this will give first 8 imgaes. labels will be numbers from 0 to 9 for above i=mages
print(image.shape) #torch.Size([8, 1, 32, 32])
# image = image.unsqueeze(0).to(device)  # shape [1, 1, 28, 28]
print(image.shape)

mask = torch.ones_like(image)
# mask[:, :, 10:28, 10:28] = 0  # mask center region
mask[:, :, :, 16:] = 0 
# def repaint(model, diffusion, x, mask, steps=1000):
def repaint(model, diffusion, x, mask, steps=100, jump_length=5, jump_n_sample=5):
    B = x.size(0)
    device = x.device
    x_t = torch.randn_like(x)  # start from pure noise

    for t_idx in reversed(range(steps)):
        t = torch.full((B,), t_idx, dtype=torch.long, device=device)

        # Predict the denoised image x0 or v or eps
        # with torch.no_grad():
        #     model_output = model(x_t, t)

        # One step of DDPM (denoise)
        out = diffusion.p_sample(
            model=model,
            x=x_t,
            t=t,
            model_kwargs={},
            clip_denoised=True,
        )

        # Replace known (unmasked) regions with original
        # x_t = out["sample"]
        # x_t = x_t * mask + x * (1 - mask)
        x_t = out["sample"] * (1 - mask) + x * mask

         # forward jump
        if (t_idx % jump_length == 0) and (t_idx > 0):
            for _ in range(jump_n_sample):
                noise = torch.randn_like(x_t)
            
                x_t = diffusion.q_sample(x_start=x_t, t=t, noise=noise)
               
                x_t = x_t * (1 - mask) + x * mask

    return x_t
# for i in range(image.size(0)):
#     output = repaint(model, diffusion, image, mask)z
output = repaint(model, diffusion, image, mask, steps=100, jump_length=10, jump_n_sample=10)
import matplotlib.pyplot as plt

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