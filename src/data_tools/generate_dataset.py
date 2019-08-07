import os
import tqdm
import glob
import cv2
import multiprocessing
import torch
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--source_path", type=str)
parser.add_argument("--target_path", default="data/celeba-HQ", type=str)
args = parser.parse_args()
os.makedirs(args.target_path, exist_ok=True)

MIN_BBOX_SIZE = 128
TARGET_IMSIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def pool(img):
  img = img.astype(np.float32)
  img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
  img = img.astype(np.uint8)
  return img

def read_image(impath):
  im = np.load(impath)[0]
  im = np.moveaxis(im, 0, 2)
  assert im.shape == (1024, 1024, 3)
  return im

def extract_and_save_image_batch(impaths, batch_idx, target_dir):

    to_save = [ [] for imsize in TARGET_IMSIZES]

    for impath in impaths:
      im = read_image(impath)
      
      for imsize_idx in range(len(TARGET_IMSIZES)-1, -1, -1):
        imsize = TARGET_IMSIZES[imsize_idx]
        
        assert im.shape[0] == imsize, f'Expected imsize: {imsize}, but got:{im.shape}'
        to_save[imsize_idx].append(im)
        im = pool(im)
    
    for imsize_idx in range(len(TARGET_IMSIZES)):
      imsize = TARGET_IMSIZES[imsize_idx]
      images = to_save[imsize_idx]
      images = np.stack(images)
      exp_shape = (imsize, imsize, 3)
      assert images.shape[1:] == exp_shape, f"Expected shape: {exp_shape}, got: {images.shape}"
      imsize_dir = os.path.join(target_dir, str(imsize))
      os.makedirs(imsize_dir, exist_ok=True)
      target_path = os.path.join(imsize_dir, f'{batch_idx}.npy')
      np.save(target_path, images)

def main(source_path, target_path):
    impaths = glob.glob(os.path.join(source_path, "*.npy"))
    print("Total number of images:", len(impaths))
    num_jobs = 1000
    batch_size = math.ceil(len(impaths) / num_jobs)
    jobs = []
    for i in range(num_jobs):
        impath = impaths[i*batch_size:(i+1)*batch_size]
        extract_and_save_image_batch(impath, i, target_path)

if __name__ == "__main__":
    main(args.source_path, args.target_path)