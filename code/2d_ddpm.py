

# %%
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Denoising Diffusion Probabilistic Models with MedNIST Dataset
#
# This tutorial illustrates how to use MONAI for training a denoising diffusion probabilistic model (DDPM)[1] to create
# synthetic 2D images.
#
# [1] - Ho et al. "Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2006.11239
#
#
# ## Setup environment

# %%
# Need to install
# monai
# monai-generative
# matplotlib

# %% [markdown]
# ## Setup imports

# %% jupyter={"outputs_hidden": false}
import os, sys
import shutil
import tempfile
import time
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

def construct_model():

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )

    return model

def construct_paths(root_dir):
    
    model_path = None
    data_path = None
    output_path = None

    if os.path.exists(root_dir):
        model_path = os.path.join(root_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        data_path = os.path.join(root_dir, "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        output_path = os.path.join(root_dir, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        print(f"Warning: root dir {root_dir} does not exist!")

    return {"model_path": model_path, "data_path": data_path, "output_path": output_path}

def construct_transform(mode = "train"):

    # Here we use transforms to augment the training dataset:
    #
    # 1. `LoadImaged` loads the hands images from files.
    # 1. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
    # 1. `ScaleIntensityRanged` extracts intensity range [0, 255] and scales to [0, 1].
    # 1. `RandAffined` efficiently performs rotate, scale, shear, translate, etc. together based on PyTorch affine transform.

    train_transforms = None

    if mode == "train":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                # transforms.RandAffined(
                #     keys=["image"],
                #     rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                #     translate_range=[(-1, 1), (-1, 1)],
                #     scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                #     spatial_size=[64, 64],
                #     padding_mode="zeros",
                #     prob=0.5
                # ),
            ]
        )
    if mode == "val":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ]
        )


    return train_transforms

def construct_scheduler():
    scheduler = DDPMScheduler(num_train_timesteps=1000) # Note the number of timestep n = 1000
    return scheduler


def train(root_dir, device=torch.device("cuda")):

    print_config()

    # %% [markdown]
    # ## Setup paths
    paths = construct_paths(root_dir)
    model_path = paths["model_path"]
    data_path = paths["data_path"]
    output_path = paths["output_path"]

    # %% [markdown]
    # ## Set deterministic training for reproducibility

    # %% jupyter={"outputs_hidden": false}
    set_determinism(42)

    # %% [markdown]
    # ## Setup MedNIST Dataset and training and validation dataloaders
    # (https://docs.monai.io/en/stable/apps.html#monai.apps.MedNISTDataset). In order to train faster, we will select just

    # %% jupyter={"outputs_hidden": false}
    if not os.path.exists(os.path.join(root_dir, "MedNIST")):
        train_data = MedNISTDataset(root_dir=data_path, section="training", download=True, progress=False, seed=0)
    else:
        print("MedNIST data already downloaded locally, skipping data download and extraction. Loading data.")
        train_data = MedNISTDataset(root_dir=data_path, section="training", download=False, progress=False, seed=0)
    # Define the directory and classes

    # Available classes: ["AbdomenCT", "BreastMRI", "CXR", "Hand", "HeadCT", "ChestCT"]
    train_datalist = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "BreastMRI"]

    # %% [markdown]
    
    # %% jupyter={"outputs_hidden": false}
    train_transforms = construct_transform(mode="train")
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)

    # Adjust batch_size based on GPU capacity, and num_workers based on cpu capacity
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)

    val_data = MedNISTDataset(root_dir=root_dir, section="validation", download=True, progress=False, seed=0)

    val_datalist = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Hand"]
    val_transforms = construct_transform(mode="val")
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)

    # %% [markdown]
    # ### Define network, scheduler, optimizer, and inferer
    # At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using
    # the original DDPM scheduler containing 1000 timesteps in its Markov chain, and a 2D UNET with attention mechanisms
    # in the 2nd and 3rd levels, each with 1 attention head.

    model = construct_model().to(device)
    scheduler = construct_scheduler()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)
    # %% [markdown]
    # ### Model training
    n_epochs = 75 # More epochs will likely result in overfitting since the MedNIST dataset is small
    val_interval = 1 # 5
    epoch_loss_list = []
    val_epoch_loss_list = []

    scaler = GradScaler()
    total_start = time.monotonic()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            print(f"training image shape is {images.shape}")
            print("haha")
            return 
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                print(f"training image shape is {images.shape}")
                print("haha")
                return 
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            noise = torch.randn((1, 1, 64, 64))
            noise = noise.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)

            plt.figure(figsize=(2, 2))
            plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(os.path.join(data_path, "plot", f"epoch_{epoch}_sample.jpg"))
            plt.close()

            # save weights
            torch.save(model.state_dict(), os.path.join(model_path, "model_weights.pth"))


        try:
            plt.style.use("seaborn-v0_8")
            plt.title("Learning Curves", fontsize=20)
            plt.plot(np.linspace(1, len(epoch_loss_list), len(epoch_loss_list)), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
            plt.plot(
                np.linspace(val_interval, len(val_epoch_loss_list), int(len(val_epoch_loss_list) / val_interval)),
                val_epoch_loss_list,
                color="C1",
                linewidth=2.0,
                label="Validation",
            )
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel("Epochs", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.legend(prop={"size": 14})
            plt.savefig(os.path.join(data_path, "plot", f"epoch_versus_loss_epoch_{epoch}.jpg"))
            plt.close()
        except:
            print("something goes wrong when saving the loss plot.")
                
    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")

    return
# %% [markdown]
# ### Learning curves

def experiment(root_dir, 
               weight_name=None,
               num_inference_steps=1000,
               replace_image_path=None,
               replace_at_t=-1, 
               device=torch.device("cuda")):
    # %% [markdown]
    # ### Plotting sampling process along DDPM's Markov chain

    from custom_inferer import CustomDiffusionInferer
    from utils.format_images import image_to_tensor

    paths = construct_paths(root_dir)
    model_path = paths["model_path"]
    data_path = paths["data_path"]
    output_path = paths["output_path"]

    plot_path = os.path.join(output_path, "eval")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # load model
    model = construct_model()
    if weight_name is not None:
        state_dict = torch.load(os.path.join(model_path, weight_name))
    else:
        state_dict = torch.load(os.path.join(model_path, "model_weights.pth"))
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)

    scheduler = construct_scheduler()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    inferer = CustomDiffusionInferer(scheduler)

    replace_tensor = None
    replace_at_t = -1
    if replace_image_path is not None:
        input_dict = {"image": replace_image_path}
        replace_tensor = construct_transform(mode="val")(input_dict)["image"].unsqueeze(0).to(device)

    with autocast(enabled=True):
        image, intermediates = inferer.sample(
            input_noise=noise, diffusion_model=model, 
            scheduler=scheduler, save_intermediates=True, 
            intermediate_steps=100,
            replace_from_t=replace_at_t,
            replace_tensor=replace_tensor
        )

    chain = torch.cat(intermediates, dim=-1)

    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(plot_path, "eval_result.jpg"))
    return
    
    

def eval(root_dir, device=torch.device("cuda")):
    # %% [markdown]
    # ### Plotting sampling process along DDPM's Markov chain

    paths = construct_paths(root_dir)
    model_path = paths["model_path"]
    data_path = paths["data_path"]
    output_path = paths["output_path"]

    plot_path = os.path.join(output_path, "eval")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # load model
    model = construct_model()
    state_dict = torch.load(os.path.join(model_path, "model_weights.pth"))
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)

    scheduler = construct_scheduler()
    scheduler.set_timesteps(num_inference_steps=1000)

    inferer = DiffusionInferer(scheduler)
    with autocast(enabled=True):
        image, intermediates = inferer.sample(
            input_noise=noise, diffusion_model=model, 
            scheduler=scheduler, save_intermediates=True, 
            intermediate_steps=100
        )

    chain = torch.cat(intermediates, dim=-1)

    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(plot_path, "eval_result.jpg"))
    return


if __name__ == "__main__":

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

    # # train model
    # train(root_dir)

    # # evaluate model
    # eval(root_dir)
    experiment(root_dir=root_dir, 
               weight_name=None,
               num_inference_steps=1000,
               replace_image_path="/isi/git/US_Diffusion_Speckle_Removal/data/BreastUS/img_0.jpeg",
               replace_at_t=500) # insert US image at halfway-point of reverse diffusion process