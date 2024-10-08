# Documentation Guide

## Prerequisites
You may have to set up a clean environment (e.g. with Conda) and use setuptools from the parent directory:
```
conda create -n lightly python=3.7
conda activate lightly
python setup.py install
pip install -r ../requirements/dev.txt
pip install -r ../requirements/openapi.txt
```

For building docs with python files (including tutorials) install detectron2.
This isn't handled in requirements because the version you'll need depends on your GPU/ hardware.
[Follow instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)


###  datastes
Imagenet2012：https://image-net.org/download.php
Imagenet100：https://image-net.org/download.php
imagenette： https://github.com/fastai/imagenette
cifar10： https://www.cs.toronto.edu/~kriz/cifar.html

### run

```
python my_BCVR_imagenet_resnet50.py
python my_BCVR_imagenet100.py
python my_BCVR_imagenette.py
python my_BCVR_cifar10.py
```

### PRETRAIN and Downstream task

```
python BCVR_pretrain_detectron2.py
python train_bcvr_detection2-coco.py
python train_bcvr_detection2-voc.py
```


### comparison
```
python benchmarks/xx_benchmark.py
python benchmarks/imagenet/resnet50/main.py
python benchmarks/imagenet/vitb16/main.py
```



