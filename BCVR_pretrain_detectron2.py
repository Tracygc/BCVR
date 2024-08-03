# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
import torch
from detectron2.detectron2 import config, modeling
from detectron2.detectron2.checkpoint import DetectionCheckpointer

from lightly.data import LightlyDataset
# from lightly.loss import NTXentLoss
from lightly.models.modules import BCVRProjectionHead
from lightly.transforms import SwaVTransform
from typing import List, Tuple
from lightly.utils import scheduler
from lightly.utils.scheduler import cosine_schedule
from lightly.models import ResNetGenerator, modules, utils
# %%
# Configuration
# -------------
# Let's set the configuration parameters for our experiments.
#
# We use a batch size of 512 and an input size of 128 in order to fit everything
# on the available amount of memory on our GPU (16GB). The number of features
# is set to the default output size of the ResNet50 backbone.
#
# We only train for 5 epochs because the focus of this tutorial is on the
# integration with Detectron2.

num_workers = 8
batch_size = 512
input_size = 128
num_ftrs = 2048

seed = 1
max_epochs = 5

# use cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# You might have downloaded the dataset somewhere else or are using a different one.
# Set the path to the dataset accordingly. Additionally, make sure to set the
# path to the config file of the Detectron2 model you want to use.
# We will be using an RCNN with a feature pyramid network (FPN).
# data_path = "/datasets/freiburg_groceries_dataset/images"
data_path = "G:/datasets/Imagenet2012/Imagenet2012/ILSVRC2012_img_val/"
cfg_path = "./Base-RCNN-FPN.yaml"
batch_size = 128
lr_factor = batch_size / 256 

# %%
# Initialize the Detectron2 Model
# --------------------------------
# The keys correspond to the different stages of the ResNet. In this tutorial, we are only
# interested in the high-level abstractions from the last layer, `res5`. Therefore,
# we have to add an additional layer which picks the right output from the dictionary.
class SelectStage(torch.nn.Module):
    """Selects features from a given stage."""

    def __init__(bcvr_backbone, stage: str = "res5"):
        super().__init__()
        bcvr_backbone.stage = stage

    def forward(bcvr_backbone, x):
        return x[bcvr_backbone.stage]


# %%
# Let's load the config file and make some adjustments to ensure smooth training.
cfg = config.get_cfg()
cfg.merge_from_file(cfg_path)

# use cuda if possible
cfg.MODEL.DEVICE = device

# randomly initialize network
cfg.MODEL.WEIGHTS = ""

# detectron2 uses BGR by default but pytorch/torchvision use RGB
cfg.INPUT.FORMAT = "RGB"

# %%
# Next, we can build the Detectron2 model and extract the ResNet50 backbone as
# follows:

detmodel = modeling.build_model(cfg)

bcvr_backbone = torch.nn.Sequential(
    detmodel.backbone.bottom_up,
    SelectStage("res5"),
    # res5 has shape bsz x 2048 x 4 x 4
    torch.nn.AdaptiveAvgPool2d(1),
).to(device)

# %%
# Finally, let's build SimCLR around the backbone as shown in the other
# tutorials. For this, we only require an additional projection head.
projection_head = BCVRProjectionHead(
    input_dim=num_ftrs,
    hidden_dim=num_ftrs,
    output_dim=128,
).to(device)

# %%
# Setup data augmentations and loaders
# ------------------------------------
#
# We start by defining the augmentations which should be used for training.
# We use the same ones as in the SimCLR paper but change the input size and
# minimum scale of the random crop to adjust to our dataset.
#
# We don't go into detail here about using the optimal augmentations.
# You can learn more about the different augmentations and learned invariances
# here: :ref:`lightly-advanced`.

CROP_COUNTS: Tuple[int, int] = (2, 6)
transform = SwaVTransform(crop_counts=CROP_COUNTS)

dataset_train = LightlyDataset(input_dir=data_path, transform=transform)

dataloader_train= torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

# %%
# bcvr_backbone-supervised pre-training
# -----------------------------
# Now all we need to do is define a loss and optimizer and start training!

# criterion = NTXentLoss()
criterion1 = NegativeCosineSimilarity()
criterion2 = bcvrLoss2()
criterion3 = VICRegLoss()
criterion4 = bcvrLoss4()
# optimizer = torch.optim.Adam(
#     list(bcvr_backbone.parameters()) + list(projection_head.parameters()),
#     lr=1e-4,
# )
warmup_epochs = 40 if max_epochs >= 800 else 20
optim = torch.optim.SGD(
    bcvr_backbone.parameters(),
    lr=0.3 * lr_factor,
    weight_decay=1e-4,
    momentum=0.9,
)
cosine_scheduler = scheduler.CosineWarmupScheduler(
    optim, warmup_epochs, max_epochs
)


for e in range(max_epochs):
    mean_loss = 0.0
    for (x0, x1), _, _ in dataloader_train:

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops= (x0, x1)
        multi_crop_features = [bcvr_backbone.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        x0= x0[0]
        x1 = x1[1]
        x0 = x0.to(bcvr_backbone.device)
        x1 = x1.to(bcvr_backbone.device)

        momentum = cosine_schedule(bcvr_backbone.current_epoch, max_epochs, 0.996, 1)  # Tico参数
        # momentum = cosine_schedule(bcvr_backbone.current_epoch, max_epochs, 0.99, 1)  # BYOL的参数
        utils.update_momentum(bcvr_backbone.backbone, bcvr_backbone.backbone_momentum, m=momentum)
        utils.update_momentum(
            bcvr_backbone.projection_head, bcvr_backbone.projection_head_momentum, m=momentum
        )

        p0 = bcvr_backbone.forward(x0)
        z0 = bcvr_backbone.forward_momentum(x0)
        p1 = bcvr_backbone.forward(x1)
        z1 = bcvr_backbone.forward_momentum(x1)

        loss = ((0.25 * (bcvr_backbone.criterion1(p0, z1) + bcvr_backbone.criterion1(p1, z0)) +
                 0.25 * bcvr_backbone.criterion2(p0, z1) +
                 0.25 * (bcvr_backbone.criterion3(p0, p1) - bcvr_backbone.criterion3(z0, z1)) +
                 0.25 * bcvr_backbone.criterion4(high_resolution_features, low_resolution_features)))
        loss.backward()

        optim.step()
        optim.zero_grad()

        # update average loss
        mean_loss += loss.detach().cpu().item() / len(dataloader_train)

    print(f"[Epoch {e:2d}] Mean Loss = {mean_loss:.2f}")


# %%
# Storing the checkpoint
# -----------------------
# Now, we can use the pre-trained backbone from the Detectron2 model. The code
# below shows how to save it as a Detectron2 checkpoint called `my_model.pth`.

# get the first module from the backbone (i.e. the detectron2 ResNet)
# backbone:
#     L ResNet50
#     L SelectStage
#     L AdaptiveAvgPool2d
detmodel.backbone.bottom_up = bcvr_backbone[0]

checkpointer = DetectionCheckpointer(detmodel, save_dir="./")
checkpointer.save("my_model")


# %%
# Finetuning with Detectron2
# ---------------------------
#
# The checkpoint from above can now be used by any Detectron2 script. For example,
# you can use the `train_net.py` script in the Detectron2 `tools`:
#
#

# %%
# .. code-block:: none
#
#   python train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#       MODEL.WEIGHTS path/to/my_model.pth \
#       MODEL.PIXEL_MEAN 123.675,116.280,103.530 \
#       MODEL.PIXEL_STD 58.395,57.120,57.375 \
#       INPUT.FORMAT RGB
#
