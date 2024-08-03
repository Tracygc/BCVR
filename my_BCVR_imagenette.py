import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import numpy as np
from lightly.utils import scheduler
from typing import List, Tuple
import copy
import os
import time
from lightly.models import modules, utils
from lightly.models.modules import heads
from lightly.utils.benchmarking import BenchmarkModule
from lightly.transforms import (
    BYOLTransform,
    DINOTransform,
    FastSiamTransform,
    SimCLRTransform,
    SimSiamTransform,
    SwaVTransform,
)
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE

from lightly.models.utils import deactivate_requires_grad

from copy import deepcopy
from lightly.data import LightlyDataset
from lightly.utils.scheduler import cosine_schedule

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

num_workers = 0
memory_bank_size = 4096

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
# knn_k = 200
knn_k = 20
knn_t = 0.1
classes = 10
input_size = 128  # input images to 128 pixels

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
# batch_size = 128
# lr_factor = batch_size / 128  # scales the learning rate linearly with batch size

batch_size = 256
lr_factor = batch_size / 256  # scales the learning rate linearly with batch size


# Number of devices and hardware to use for training.
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

if distributed:
    strategy = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // devices
else:
    strategy = None  # Set to "auto" if using PyTorch Lightning >= 2.0
    # limit to single device if not using distributed training
    devices = min(devices, 1)

# The dataset structure should be like this:

path_to_train = "G:datasets/imagenette/imagenette2-160/train"
path_to_test = "G:datasets/imagenette/imagenette2-160/val"

class BCVR(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # # create a ResNet backbone and remove the classification head

        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Relu++BN
        self.projection_head = heads.BCVRProjectionHead(feature_dim, 4096, 3000)
        self.prototypes = heads.SwaVPrototypes(256, 3000)  # use 3000 prototypes

        self.prediction_head = heads.BCVRPredictionHead(3000, 4096, 256)
        self.backbone_momentum = deepcopy(self.backbone)
        self.projection_head_momentum = deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.warmup_epochs = 40 if max_epochs >= 800 else 20

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        p = self.prediction_head(x)
        x = nn.functional.normalize(p, dim=1, p=2)
        return self.prototypes(x)
        # return x

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = nn.functional.normalize(z, dim=1, p=2)
        z = z.detach()
        # return self.prototypes(z)
        return z


    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        loss = 'I will publish the model code after my article is accepted'
        return loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.3 * lr_factor,
            weight_decay=1e-4,
            momentum=0.9,
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]

normalize_transform = torchvision.transforms.Normalize(
    mean=IMAGENET_NORMALIZE["mean"],
    std=IMAGENET_NORMALIZE["std"],
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        normalize_transform,
    ]
)


# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)
dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)
def get_data_loaders(batch_size: int, dataset_train_ssl):
    """Helper method to create dataloaders for ssl, kNN train and kNN test.

    Args:
        batch_size: Desired batch size for all dataloaders.
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


# Multi crop augmentation for DINO, additionally, disable blur for cifar10
transform1 = DINOTransform(
    global_crop_size=128,
    local_crop_size=64,
    cj_strength=0.5,
)
# we ignore object detection annotations by setting target_transform to return 0
# transform2 = BYOLTransform(
#     view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
#     view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
# )
transform2 = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=input_size),
    view_2_transform=BYOLView2Transform(input_size=input_size),
)

transform3 = SwaVTransform(
    crop_sizes=(128, 64),
    crop_counts=(2, 6),  # 2 crops @ 128x128px and 6 crops @ 64x64px
    cj_strength=0.5,
)

transform4 = SimSiamTransform(input_size=input_size)

# )

# or create a dataset from a folder containing images or videos:

dataset = LightlyDataset(path_to_train, transform = transform3)
# dataset = LightlyDataset(path_to_train, transform=transform3)
dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
    batch_size=batch_size, dataset_train_ssl=dataset
)
# benchmark_model = model(dataloader_train_kNN, classes)
model = BCVR(dataloader_train_kNN, classes)

# model = SwaVModel(dataloader_train_kNN, classes)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator=accelerator)

start = time.time()
trainer.fit(model=model, train_dataloaders=dataloader_train_ssl, val_dataloaders=dataloader_test,)
end = time.time()

runs = []
run = {
    "model": model,
    "batch_size": batch_size,
    "epochs": max_epochs,
    "max_accuracy": model.max_accuracy,
    "runtime": end - start,
    "gpu_memory_usage": torch.cuda.max_memory_allocated(),
    # "seed": seed,
}
runs.append(run)
print(run)

# del model
# del trainer
# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()

bench_results = dict()
bench_results[model] = runs
# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print("-" * len(header))
print(header)
print("-" * len(header))
for model, results in bench_results.items():
    runtime = np.array([result["runtime"] for result in results])
    runtime = runtime.mean() / 60  # convert to min
    accuracy = np.array([result["max_accuracy"] for result in results])
    gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3)  # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True,
    )
print("-" * len(header))
