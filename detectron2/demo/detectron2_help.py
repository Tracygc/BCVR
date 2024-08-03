import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

class Detectron2_help:
    def __init__(self):
        self.config_file = r"D:\detectron2-v0.6\configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        self.confidence_threshold = 0.5
        self.opts = ["MODEL.WEIGHTS", r"D:\detectron2-v0.6\demo\mask_rcnn_R_50_FPN_3x.pkl"]


    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(self.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.freeze()
        return cfg

    def inference_img(self,image_file):
        mp.set_start_method("spawn", force=True)
        setup_logger(name="fvcore")
        cfg = self.setup_cfg()
        demo = VisualizationDemo(cfg)

        if image_file:
            img = read_image(image_file, format="BGR")
            predictions, visualized_output, mask = demo.run_on_image(img)

        return predictions,visualized_output,mask

if __name__ == "__main__":
    de = Detectron2_help()
    predictions,visualized_output,mask = de.inference_img(r"D:\detectron2-v0.6\test_data\1.jpg")
    print("predictions",predictions)
    print("visualized_output", mask)
    print("mask", mask)
    cv2.imshow("frame",visualized_output.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
























