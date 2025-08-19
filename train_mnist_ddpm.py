# train_mnist_diffusion.py
import torch, torch.nn as nn
from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    # parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_channels', type=int, default=128)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((1,), (1))
    ])
    ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model_args = {
        'image_size': args.image_size, 'num_channels': args.model_channels,
        'num_res_blocks': 2, 'channel_mult': [1, 2, 2, 2],
        'attention_resolutions': "", 'dropout': 0.1
    }
    diffusion_args = {
        'diffusion_steps': 1000, 'noise_schedule': 'linear',
        'timestep_respacing': '1000'
    }
    # model, diffusion = create_model_and_diffusion(**{**model_args, **diffusion_args})
    model, diffusion = create_model_and_diffusion(
    image_size=32,
    num_channels=64,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    # num_classes=10,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
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


    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        for batch, _ in dl:
            # batch = batch.cuda() * 2 - 1
            batch = batch.cuda() #changed this
            t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=batch.device).long()
            loss_dict = diffusion.training_losses(model, batch, t)
            loss = loss_dict["loss"].mean()
            # loss = diffusion.training_losses(model, batch)['loss'].mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss.item():.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"mnist_ddpm_epoch{epoch+1}.pt")

        # for i, (batch, _) in enumerate(data):
        #     model.train()
        #     batch = batch.to(device)

        #     # Sample random time steps for each batch element
        #     t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=batch.device).long()

        #     # Compute training loss
        #     loss_dict = diffusion.training_losses(model, batch, t)
        #     loss = loss_dict["loss"].mean()
            
        #     # Backprop and optimize
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

if __name__ == "__main__":
    main()
