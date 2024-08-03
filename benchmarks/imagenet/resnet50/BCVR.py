import copy
from typing import List, Tuple
from copy import deepcopy

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
import torch.nn as nn
import torchvision
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.models.modules import SwaVPrototypes,BCVRProjectionHead,BCVRPredictionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.models.utils import deactivate_requires_grad
from lightly.models import modules, utils
from lightly.transforms import SwaVTransform


# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False
# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 1
CROP_COUNTS: Tuple[int, int] = (2, 6)


class BCVR(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        # resnet = resnet50()
        # resnet.fc = Identity()  # Ignore classification head
        # self.backbone = resnet
        resnet = torchvision.models.resnet50()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # BN放在ReLu之后
        self.projection_head = BCVRProjectionHead(feature_dim, 4096, 3000)
        self.prototypes = SwaVPrototypes(256, 3000)  # use 3000 prototypes
        self.prototypes = SwaVPrototypes(n_steps_frozen_prototypes=1)

        self.prediction_head = BCVRPredictionHead(3000, 4096, 256)
        self.backbone_momentum = deepcopy(self.backbone)
        self.projection_head_momentum = deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.warmup_epochs = 40 if max_epochs >= 800 else 20
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

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


    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    # def configure_optimizers(self):
    #     # 使用tico的sgd
    #     optim = torch.optim.SGD(
    #         self.parameters(),
    #         lr=0.3 * lr_factor,
    #         weight_decay=1e-4,
    #         momentum=0.9,
    #     )
    #     cosine_scheduler = scheduler.CosineWarmupScheduler(
    #         optim, self.warmup_epochs, max_epochs
    #     )
    #     return [optim], [cosine_scheduler]

    # 使用tico的sgd
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.projection_head,
            ]
        )
        optimizer = LARS(
            [
                {"name": "BCVR", "params": params},
                {
                    "name": "BCVR_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.2 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1.5e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

# bcvr 使用swav的transformer
transform = SwaVTransform(crop_counts=CROP_COUNTS)