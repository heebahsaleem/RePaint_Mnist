
#Kaggle + repaint

import sys
import os
import torch
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import os

os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/lib:/lib64:/usr/lib:/usr/lib64"

# %env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/lib64:/usr/lib:/usr/lib64

os.environ["HF_HUB_OFFLINE"] = "1"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




model_path = "./mnist_unet"
scheduler_path = "./mnist_scheduler"
class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation, and Convolution operations.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 num_groups:int = 8, 
                 kernel_size: int = 3, 
                 norm:bool = True,
                 act:bool = True
                ):
        super(NormActConv, self).__init__()
        
        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            in_channels
        ) if norm is True else nn.Identity()
        
        # Activation
        self.act = nn.SiLU() if act is True else nn.Identity()
        
        # Convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size, 
            padding=(kernel_size - 1)//2
        )
        
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """
    def __init__(self, 
                 n_out:int, # Output Dimension
                 t_emb_dim:int = 128 # Time Embedding Dimension
                ):
        super(TimeEmbedding, self).__init__()
        
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)
    

class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multiheaded Self Attention operation.    
    """
    def __init__(self, 
                 num_channels:int,
                 num_groups:int = 8, 
                 num_heads:int = 4,
                 norm:bool = True
                ):
        super(SelfAttentionBlock, self).__init__()
        
        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            num_channels
        ) if norm is True else nn.Identity()
        
        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads, 
            batch_first=True
        )
        
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x
    

class Downsample(nn.Module):
    """
    Perform Downsampling by the factor of k across Height and Width.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 k:int = 2, # Downsampling factor
                 use_conv:bool = True, # If Downsampling using conv-block
                 use_mpool:bool = True # If Downsampling using max-pool
                ):
        super(Downsample, self).__init__()
        
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        
        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_mpool else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            )
        ) if use_conv else nn.Identity()
        
        # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_mpool else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
            return self.mpool(x)
        
        if not self.use_mpool:
            return self.cv(x)
            
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)
    

class Upsample(nn.Module):
    """
    Perform Upsampling by the factor of k across Height and Width
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 k:int = 2, # Upsampling factor
                 use_conv:bool = True, # Upsampling using conv-block
                 use_upsample:bool = True # Upsampling using nn.upsample
                ):
        super(Upsample, self).__init__()
        
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        
        # Upsampling using conv
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels//2 if use_upsample else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            ),
            nn.Conv2d(
                out_channels//2 if use_upsample else out_channels, 
                out_channels//2 if use_upsample else out_channels, 
                kernel_size = 1, 
                stride=1, 
                padding=0
            )
        ) if use_conv else nn.Identity()
        
        # Upsamling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k, 
                mode = 'bilinear', 
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_upsample else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
            return self.up(x)
        
        if not self.use_upsample:
            return self.cv(x)
        
        return torch.cat([self.cv(x), self.up(x)], dim=1)

class DownC(nn.Module):
    """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 t_emb_dim:int = 128, # Time Embedding Dimension
                 num_layers:int=2,
                 down_sample:bool = True # True for Downsampling
                ):
        super(DownC, self).__init__()
        
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])
        
        self.down_block =Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, t_emb):
        
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn

        # Downsampling
        out = self.down_block(out)
        
        return out

class MidC(nn.Module):
    """
    Refine the features obtained from the DownC block.
    It refines the features using following operations:
    
    1. Resnet Block with Time Embedding
    2. A Series of Self-Attention + Resnet Block with Time-Embedding 
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 t_emb_dim:int = 128,
                 num_layers:int = 2
                ):
        super(MidC, self).__init__()
        
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers + 1)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers + 1)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])
        
    def forward(self, x, t_emb):
        out = x
        
        # First-Resnet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)
        
        # Sequence of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.conv1[i+1](out)
            out = out + self.te_block[i+1](t_emb)[:, :, None, None]
            out = self.conv2[i+1](out)
            out = out + self.res_block[i+1](resnet_input)
            
        return out

class UpC(nn.Module):
    """
    Perform Up-convolution on the input using following approach.
    1. Upsampling
    2. Conv + TimeEmbedding
    3. Conv
    4. Skip-connection from 1.
    5. Self-Attention
    6. Skip-Connection from 3.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 t_emb_dim:int = 128, # Time Embedding Dimension
                 num_layers:int = 2,
                 up_sample:bool = True # True for Upsampling
                ):
        super(UpC, self).__init__()
        
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])
        
        self.up_block =Upsample(in_channels, in_channels//2) if up_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, down_out, t_emb):
        
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        return out


def get_time_embedding(
    time_steps: torch.Tensor,
    t_emb_dim: int
) -> torch.Tensor:
    
    """ 
    Transform a scalar time-step into a vector representation of size t_emb_dim.
    
    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)
    
    :return tensor of size -> (B, t_emb_dim)
    """
    
    assert t_emb_dim%2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start = 0, 
                              end = t_emb_dim//2, 
                              dtype=torch.float32, 
                              device=time_steps.device
                             ) / (t_emb_dim)
    
    factor = 10000**factor

    t_emb = time_steps[:,None] # B -> (B, 1) 
    t_emb = t_emb/factor # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1) # (B , t_emb_dim)
    
    return t_emb

class Unet(nn.Module):
    """
    U-net architecture which is used to predict noise
    in the paper "Denoising Diffusion Probabilistic Model".
    
    U-net consists of Series of DownC blocks followed by MidC
    followed by UpC.
    """
    
    def __init__(self,
                 im_channels: int = 1, 
                 down_ch: list = [32, 64, 128, 256],
                 mid_ch: list = [256, 256, 128],
                 up_ch: list[int] = [256, 128, 64, 16],
                 down_sample: list[bool] = [True, True, False],
                 t_emb_dim: int = 128,
                 num_downc_layers:int = 2, 
                 num_midc_layers:int = 2, 
                 num_upc_layers:int = 2
                ):
        super(Unet, self).__init__()
        
        self.im_channels = im_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.up_ch = up_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers
        
        self.up_sample = list(reversed(self.down_sample)) # [False, True, True]
        
        # Initial Convolution
        self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)
        
        # Initial Time Embedding Projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim), 
            nn.SiLU(), 
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(
                self.down_ch[i], 
                self.down_ch[i+1], 
                self.t_emb_dim, 
                self.num_downc_layers, 
                self.down_sample[i]
            ) for i in range(len(self.down_ch) - 1)
        ])
        
        # MidC Block
        self.mids = nn.ModuleList([
            MidC(
                self.mid_ch[i], 
                self.mid_ch[i+1], 
                self.t_emb_dim, 
                self.num_midc_layers
            ) for i in range(len(self.mid_ch) - 1)
        ])
        
        # UpC Block
        self.ups = nn.ModuleList([
            UpC(
                self.up_ch[i], 
                self.up_ch[i+1], 
                self.t_emb_dim, 
                self.num_upc_layers, 
                self.up_sample[i]
            ) for i in range(len(self.up_ch) - 1)
        ])
        
        # Final Convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]), 
            nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        
        out = self.cv1(x)
        
        # Time Projection
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        # DownC outputs
        down_outs = []
        
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        
        # MidC outputs
        for mid in self.mids:
            out = mid(out, t_emb)
        
        # UpC Blocks
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            
        # Final Conv
        out = self.cv2(out)
        
        return out



model = Unet().to(device)
model.load_state_dict(torch.load("./mnist_unet_kaggle_model_weights/mnist_unet_kaggle.pth"))
model.eval()

# state_dict = torch.load("./mnist_unet_kaggle_model_weights/mnist_unet_kaggle.pth", map_location=device)
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000) 
alphas = scheduler.alphas_cumprod.to(device)

# model = UNet2DModel.from_pretrained(model_path).to(device)
# model.eval()
# scheduler = DDPMScheduler.from_pretrained(scheduler_path)
# alphas = scheduler.alphas_cumprod.to(device)#storing scheduler parameter


transform = transforms.Compose([
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
            noise_pred = model(x_t, t_tensor)
        

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
    output = torch.clamp(output, 0, 1) #removed sigmoid, the background turned to grey by using sigmoid
    
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
print("fdsff")
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
# plt.show()
plt.savefig('sine_wave3.png') 

#as kristian suggested to also check without masking or unconditional generation to verify model quality
#after sampling, it will test unconditional generation 
# print("this is unconditiional gneration testing")

# with torch.no_grad():
#     unconditional_noise = torch.randn(4, 1, 32, 32, device=device)
#     unconditional_samples = []
    
#     for t in reversed(range(100)):
#         t_tensor = torch.tensor([t] * 4, device=device, dtype=torch.long)
#         noise_pred = model(unconditional_noise, t_tensor)
        
#         alpha_t = scheduler.alphas_cumprod[t].to(device)
#         beta_t = 1 - alpha_t
#         print("eorking")
#         if t > 0:
#             noise = torch.randn_like(unconditional_noise)
#         else:
#             noise = torch.zeros_like(unconditional_noise)
        
#         unconditional_noise = (1 / torch.sqrt(alpha_t)) * (
#             unconditional_noise - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred
#         ) + torch.sqrt(beta_t) * noise
    
#     unconditional_output = (unconditional_noise + 1) / 2
#     unconditional_output = torch.clamp(unconditional_output, 0, 1)

# plt.figure(figsize=(10, 2.5))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(unconditional_output[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
#     plt.title("Unconditional")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()