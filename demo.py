import sys

from utils.data_viz import display_flow
from utils.file_io import save_flow_image
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import data
import models
import demo_frames
import output

model_dir = os.path.dirname(models.__file__)
model_path = os.path.join(model_dir, "raft-sintel.pth")

data_path = os.path.dirname(data.__file__)
frames_dir = os.path.join(data_path, "frames")
output_dir = os.path.dirname(output.__file__)



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(100)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    for parent_dir, sub_dirs, images in os.walk(args.data_dir):
        
        # prevent out of GPU memory error when inferring over many different image sets
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            
            images = tuple(os.path.join(parent_dir, image) for image in images)
            images = sorted(images)

            for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                
                flow_permuted = flow_up[0].permute(1,2,0).cpu().numpy()
                # flo_permuted = flow_viz.flow_to_image(flo_permuted)
                
                
                # image1 = image1[0].permute(1,2,0).cpu().numpy()
                
                #display_flow(flow_up, None, image1)
                if "sintel" not in parent_dir:
                    dir_name_for_frame_src = os.path.basename(parent_dir)
                else:
                    dir_name_for_frame_src = "sintel/market_2/final"
                output_path = os.path.join(output_dir, dir_name_for_frame_src, "RAFT")
                save_flow_image(flow_permuted, idx, output_path)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=model_path, help="restore checkpoint")
    parser.add_argument('--data_dir', default=frames_dir, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
