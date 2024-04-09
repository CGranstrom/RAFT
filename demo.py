from cmath import sin
from multiprocessing.pool import RUN
import sys
from output import opt_flow_output, opt_flow_output_640_480, scratch, system_resource_metrics_output

#from utils.data_viz import display_flow
from utils.file_io import save_flow_image
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image

#from raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
import pandas as pd
from core.raft import RAFT
from timeit import default_timer as timer


import data
import models
import demo_frames
import itertools

model_dir = os.path.dirname(models.__file__)
#model_path = os.path.join(model_dir, "raft-kitti.pth")
MODEL_PATHS = [os.path.join(model_dir, "raft-sintel.pth"), os.path.join(model_dir, "raft-kitti.pth")]



DATA_PATH = os.path.dirname(data.__file__)
FRAMES_DIR = os.path.join(DATA_PATH, "frames_640_480")
# output_dir = os.path.dirname(output.__file__)
#OUTPUT_DIR = os.path.dirname(opt_flow_output_640_480.__file__)
#METRICS_REPORT_DIR = os.path.join(DATA_PATH, "../system_resource_metrics")

FLOW_OUTPUT_DIR = os.path.dirname(scratch.__file__)
METRICS_REPORT_DIR = os.path.dirname(system_resource_metrics_output.__file__)

#RUN_MODE = "perf_testing"
RUN_MODE = "gen_images"
NUM_IMAGES_PER_DIR = 5
clock_types = ["wall"]




DEVICE = 'cuda'

run_stats = pd.DataFrame

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    for model_path, CLOCK_TYPE in itertools.product(MODEL_PATHS, clock_types):
        print(f"NOW ON MODEL {model_path}, clock type {CLOCK_TYPE}\n\n\n\n\n")
        
        experiment_label = f"TOTAL for {os.path.basename(model_path)} {CLOCK_TYPE} time"
        dir_names = [experiment_label]

        for parent_dir, _, images in sorted(os.walk(args.data_dir)):  # default is FRAMES_DIR
            
            if not images:
                continue
            if "sintel" in parent_dir:
                dir_names.append("sintel")
            else:
                dir_names.append(os.path.basename(parent_dir))


        #dir_names.append(video_dir)
        
        run_stats = pd.DataFrame(
        [[0] * 5 + [(1, 1)]] * len(dir_names),
        columns=[
            "dir_agg_runtime",
            "dir_mean_runtime",
            "dir_median_runtime",
            "dir_std_dev_runtime",
            "dir_num_trials",
            "image_res",
            ],
        index=dir_names,
        )
        total_runtimes = []
        
        TRIAL_CTR=0            

        for parent_dir, sub_dirs, images in sorted(os.walk(args.data_dir)):  #default is FRAMES_DIR
            
            # prevent out of GPU memory error when inferring over many different image sets
            torch.cuda.empty_cache()
            
            if RUN_MODE == "perf_testing":
                    
                with torch.no_grad():
                    
                    

                    if not images:
                        continue                 
                    images = tuple(os.path.join(parent_dir, image) for image in images)
                    images = sorted(images)
                    images = images[:NUM_IMAGES_PER_DIR + 1]
                    
                    (dir_num_trials,dir_agg_runtime,dir_mean_runtime,dir_median_runtime,dir_std_dev_runtime) = (0, 0, 0, 0, 0)
                        
                    dir_runtimes = []

                    for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                        image1 = load_image(imfile1)
                        image2 = load_image(imfile2)

                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)
                        
                        image_res = cv2.imread(os.path.join(parent_dir, images[0])).shape[:2]
                        

                        
                        TRIAL_CTR += 1
                        print(f"TRIAL_CTR={TRIAL_CTR}\n\n\n\n\n")
                        
                        #yappi.start()
                        start = timer()
                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        end = timer()
                        elapsed_time = end-start
                        #yappi.stop()
                        
                        # get total runtime for model()
                        # stats = yappi.get_func_stats(
                        #     filter_callback=lambda x: yappi.func_matches(
                        #         x, [RAFT.forward]
                        #     )
                        # )
                        
                        
                        #dir_runtimes.append(list(stats._as_dict.keys())[0][6])
                        dir_runtimes.append(elapsed_time)
                        #yappi.clear_stats()
                        
                        #flow_permuted = flow_up[0].permute(1,2,0).cpu().numpy()
                        
                        #display_flow(flow_up, None, image1)
                        if "sintel" not in parent_dir:
                            dir_name_for_frame_src = os.path.basename(parent_dir)
                        else:
                            dir_name_for_frame_src = "sintel/market_2/final"
                        
                        # output_path = os.path.join(OUTPUT_DIR, dir_name_for_frame_src, "RAFT_raft-kitti")
                        
                        #save_flow_image(flow_permuted, idx, output_path, res=(640, 480))
                    
                    dir_agg_runtime = sum(dir_runtimes)
                    dir_mean_runtime = np.mean(dir_runtimes)
                    dir_median_runtime = np.median(dir_runtimes)
                    dir_std_dev_runtime = np.std(dir_runtimes)
                    dir_num_trials = len(dir_runtimes)

                    if "sintel" in dir_name_for_frame_src:
                        dir_name_for_frame_src = "sintel"

                    # run_stats.loc[dir_name_for_frame_src] = {
                    #     "dir_agg_runtime": dir_agg_runtime,
                    #     "dir_mean_runtime": dir_mean_runtime,
                    #     "dir_median_runtime": dir_median_runtime,
                    #     "dir_std_dev_runtime": dir_std_dev_runtime,
                    #     "dir_num_trials": dir_num_trials,
                    #     "image_res": image_res,
                    # }
                    tmp_dict = {"dir_agg_runtime": dir_agg_runtime, "dir_mean_runtime": dir_mean_runtime, "dir_median_runtime": dir_median_runtime, "dir_std_dev_runtime":dir_std_dev_runtime, "dir_num_trials": dir_num_trials, "image_res": image_res}
        
                    run_stats.loc[dir_name_for_frame_src, tmp_dict.keys()] = tmp_dict.values()
                    total_runtimes.extend(dir_runtimes)
                    
                    #yappi.clear_stats()


            
            elif RUN_MODE == "gen_images":
                
                with torch.no_grad():
                
                    images = tuple(os.path.join(parent_dir, image) for image in images)
                    images = sorted(images)
                    images = images[:NUM_IMAGES_PER_DIR + 1]

                    for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                        image1 = load_image(imfile1)
                        image2 = load_image(imfile2)

                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)

                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        
                        flow_permuted = flow_up[0].permute(1,2,0).cpu().numpy()
                        
                        #display_flow(flow_up, None, image1)
                        dir_name_for_frame_src = os.path.basename(parent_dir)
                        output_path = os.path.join(FLOW_OUTPUT_DIR, dir_name_for_frame_src, f"RAFT_{os.path.basename(model_path)}")
                        save_flow_image(flow_permuted, idx, output_path, res=(640, 480))
        
        #uncomment below for runtime stats
        
        # total_runtime = sum(total_runtimes)
        # total_mean_runtime = np.mean(total_runtimes)
        # total_median_runtime = np.median(total_runtimes)
        # total_std_dev_runtime = np.std(total_runtimes)
        # total_num_trials = len(total_runtimes)
        
        # tmp_dict = {"dir_agg_runtime": total_runtime, "dir_mean_runtime": total_mean_runtime, "dir_median_runtime": total_median_runtime, "dir_std_dev_runtime":total_std_dev_runtime, "dir_num_trials": total_num_trials, "image_res": np.nan}
        
        # run_stats.loc[experiment_label, tmp_dict.keys()] = tmp_dict.values()

        # # run_stats.loc[experiment_label] = {
        # #     "dir_agg_runtime": total_runtime,
        # #     "dir_mean_runtime": total_mean_runtime,
        # #     "dir_median_runtime": total_median_runtime,
        # #     "dir_std_dev_runtime": total_std_dev_runtime,
        # #     "dir_num_trials": total_num_trials,
        # #     "image_res": np.nan,
        # #     }
        
        
        # run_stats.to_csv(
        #     os.path.join(
        #         METRICS_REPORT_DIR,
        #         f"{os.path.basename(model_path)}_{CLOCK_TYPE}_runtimes_640_480_timer_not_yappi.csv",
        #     )
        # )    

if __name__ == '__main__':
    
    class IgnorantActionsContainer(argparse._ActionsContainer):
        def _handle_conflict_ignore(self, action, conflicting_actions):
            pass

    argparse.ArgumentParser.__bases__ = (argparse._AttributeHolder, IgnorantActionsContainer)
    argparse._ArgumentGroup.__bases__ = (IgnorantActionsContainer,)

    parser = argparse.ArgumentParser(conflict_handler="ignore")
    
    for model_path in MODEL_PATHS:
        parser.add_argument('--model', default=model_path, help="restore checkpoint")
        parser.add_argument('--data_dir', default=FRAMES_DIR, help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()

        demo(args)
