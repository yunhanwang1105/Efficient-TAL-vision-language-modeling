import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import itertools,operator
from stale_model_single_input import STALE
import stale_lib.stale_dataloader as stale_dataset
from scipy import ndimage
from scipy.special import softmax
from collections import Counter
import cv2
import json
from config.dataset_class import activity_dict
import yaml
from utils.postprocess_utils import multithread_detection , get_infer_dict, load_json
from joblib import Parallel, delayed
from config.dataset_class import activity_dict
from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test
from fvcore.nn import FlopCountAnalysis, flop_count_table,ActivationCountAnalysis
import time

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

def get_mem_usage():
    GB = 1024.0 ** 3
    num_gpu = config['training']['num_gpu']
    output = ["device_%d = %.03fGB" % (device, torch.cuda.max_memory_allocated(torch.device('cuda:%d' % device)) / GB) for device in range(num_gpu)]
    return ' '.join(output)[:-1]

if __name__ == '__main__':

    output_path = config['dataset']['testing']['output_path']
    pretrain_mode = config['model']['clip_pretrain']
    split = config['dataset']['split']

    rand_seeds = [0, 100, 200, 300, 400]
    vid_lengths = np.concatenate(([100], np.arange(200,3200,200)), axis=0) 

    all_results = np.zeros((len(rand_seeds), len(vid_lengths)))

    model_saved = False

    for experiment_index, rand_seed in enumerate(rand_seeds): # 5 experiments

        torch.manual_seed(rand_seed)  # set the seed for reproducibility

        secs_list = []
        for v_index, v_length in enumerate(vid_lengths):

            
            ### Load Model ###
            model = STALE(temporal_scale = v_length)

            # Enable quantization

            model = torch.quantization.quantize_dynamic(
                        model, {nn.Linear, nn.Conv1d, nn.Conv2d}, dtype=torch.qint8
                    )
            
            if not model_saved:
                torch.save(model, "quantized_model.pth.tar")
                model_saved = True

            model.eval()

            input_data = torch.rand(1, 512, v_length).to('cpu')
            test_loader = [input_data]

            # Test time
            start_time= time.time()

            model(input_data)

            stop_time=time.time()
            duration =stop_time - start_time
            hours = duration // 3600
            minutes = (duration - (hours * 3600)) // 60
            seconds = duration - ((hours * 3600) + (minutes * 60))
            # print("Forward time: "+ str(seconds) + " secs")  # milliseconds

            rounded_secs = round(seconds, 3)
            all_results[experiment_index][v_index] = rounded_secs

            del model

    mean_secs = np.mean(all_results, axis=0)
    std_secs = np.std(all_results, axis=0)
    print("The mean secs list is: ", mean_secs)
    print()
    print("The std secs list is: ", std_secs)

        
            

