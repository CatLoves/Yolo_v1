import torch
import torch.nn as nn
import pandas as pd
import os
import PIL
import skimage
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from tqdm import tqdm
seed = 123
import cv2
import xml.etree.ElementTree as ET
torch.manual_seed(seed)
from collections import Counter
from utils import *

from model_YOLOV1 import YoloV1
from loss import YoloLoss
from dataset import FruitImagesDataset, test_dir, test_df

LOAD_MODEL = True
EPOCHS = 1
LOAD_MODEL_FILE = '../output/model.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def predictions():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0
    )
    loss_fn = YoloLoss()
    BATCH_SIZE = 1

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        print(f"")

    test_dataset = FruitImagesDataset(
        transform=transform, 
        df=test_df,
        files_dir=test_dir
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
        
    for epoch in range(EPOCHS):

        model.eval()
        
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Test mAP: {mean_avg_prec}")


predictions()