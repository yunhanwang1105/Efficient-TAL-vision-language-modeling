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

    is_postprocess = True
    if not os.path.exists(output_path + "/results"):
        os.makedirs(output_path + "/results")


    vid_lengths = vid_lengths = np.concatenate(([100], np.arange(200,3200,200)), axis=0) 

    for v_length in vid_lengths:
        print()
        print("##################### ", v_length, " #####################")
        ### Load Model ###

        model = STALE(temporal_scale = v_length)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda() 
        ### Load Checkpoint ###
        #checkpoint = torch.load(output_path + "STALE_best_100p2.pth.tar")
        #model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        ### Load Dataloader ###
        # test_loader = torch.utils.data.DataLoader(stale_dataset.STALEDataset(subset="validation", mode='inference'),
        #                                           batch_size=1, shuffle=False,
        #                                           num_workers=4, pin_memory=True, drop_last=False)

        input_data = torch.zeros(1, 512, v_length).to('cuda')
        test_loader = [input_data]

        # Test memory
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        model(input_data)
        end.record()
        torch.cuda.synchronize()
        print("Forward time :"+ str(start.elapsed_time(end)/1000) + " secs")  # milliseconds

        memo = get_mem_usage()
        print("Memory usage info: ", memo)

        # Test FLOPs
        flops = FlopCountAnalysis(model, input_data)
        gmacs = flops.total() / (10 ** 9)
        print("GMACs = ", gmacs)
        

        # split = (100 - config['dataset']['split'])

        # if split == 50 :
        #     class_to_idx = t2_dict_test
        # elif split == 25 :
        #     class_to_idx = t1_dict_test


        # key_list = list(class_to_idx.keys())
        # val_list = list(class_to_idx.values())

        key_list = list(activity_dict.keys())
        val_list = list(activity_dict.values())

        nms_thres = config['testing']['nms_thresh']
        
        #file = "stale_output.csv"
        #if(os.path.exists(file) and os.path.isfile(file)):
        #    os.remove(file)

        # # print("Inference start")
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # with torch.no_grad():
        #     vid_count=0
        #     match_count=0
        #     vid_label_dict = {}
        #     results = {}
        #     result_dict = {}
        #     # splits = (100 - config['dataset']['split']) / 100
        #     # class_thres = config['testing']['cls_thresh']
        #     num_class = config['dataset']['num_classes']
        #     # num_class = int(splits*config['dataset']['num_classes'])
        #     # num_class = 50
        #     top_k_snip = config['testing']['top_k_snip']
        #     class_snip_thresh = config['testing']['class_thresh']
        #     mask_snip_thresh = config['testing']['mask_thresh']
        #     tscale = config['model']['temporal_scale']
            

        #     new_props = list()

        #     for input_data in test_loader:

        #         video_name = "random_video"
        #         vid_count+=1
        #         input_data = input_data.cuda()
        #         top_br_pred, bottom_br_pred,_,cl_pred,_ = model(input_data)
        #         # forward pass
        #         ### global mask prediction ####
        #         props = bottom_br_pred[0].detach().cpu().numpy()

        #         ### classifier branch prediction ###
        #         soft_cas = torch.softmax(top_br_pred[0],dim=0) 

        #         cl_pred = torch.softmax(cl_pred[0],dim=0)
        #         cl_pred_np =  cl_pred.detach().cpu().numpy()
        #         soft_cas_topk,soft_cas_topk_loc = torch.topk(soft_cas[:num_class],2,dim=0)
        #         top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class]
        #         label_pred = torch.softmax(torch.mean(top_br_pred[0][:num_class,:],dim=1),axis=0).detach().cpu().numpy()
        #         vid_label_id = np.argmax(label_pred)
        #         vid_label_sc = np.amax(label_pred)
        #         props_mod = props[props>0]
        #         top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class]
        #         top_br_mean = np.mean(top_br_np,axis=1)
        #         top_br_mean_max = np.amax(top_br_np,axis=1)
        #         top_br_mean_id = np.argmax(top_br_mean)
        #         soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
        #         seg_score = np.zeros([tscale])
        #         seg_cls = []
        #         seg_mask = np.zeros([tscale])

        #         ### for each snippet, store the max score and class info ####

        #         for j in range(tscale):
                    
        #             seg_score[j] =  np.amax(soft_cas_np[:,j])
        #             seg_cls.append(np.argmax(soft_cas_np[:,j]))

        #         thres = class_snip_thresh

        #         cas_tuple = []
        #         for k in thres:
        #             filt_seg_score = seg_score > k
        #             integer_map1 = map(int,filt_seg_score)
        #             filt_seg_score_int = list(integer_map1)
        #             filt_seg_score_int = ndimage.binary_fill_holes(filt_seg_score_int).astype(int).tolist()
        #             if 1 in filt_seg_score_int:
        #                 start_pt1 = filt_seg_score_int.index(1)
        #                 end_pt1 = len(filt_seg_score_int) - 1 - filt_seg_score_int[::-1].index(1)
        #                 if end_pt1 - start_pt1 > 1:
        #                     scores = np.amax(seg_score[start_pt1:end_pt1])
        #                     label = max(set(seg_cls[start_pt1:end_pt1]), key=seg_cls.count)
        #                     cas_tuple.append([start_pt1,end_pt1,scores,label])

        #         max_score, score_idx  = torch.max(soft_cas[:num_class],0)
        #         soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
        #         score_map = {}
        #         top_np = top_br_pred[0][:num_class].detach().cpu().numpy()  
        #         top_np_max = np.mean(top_np,axis=1)
        #         max_score_np = max_score.detach().cpu().numpy()
        #         score_idx = score_idx.detach().cpu().numpy()

        #         for ids in range(len(score_idx)):
        #             score_map[max_score_np[ids]]= score_idx[ids]

        #         #print(score_idx)
        #         #print(score_map)

        #         k = top_k_snip ## more fast inference
        #         max_idx = np.argpartition(max_score_np, -k)[-k:]

        #         ### indexes of top K scores ###

        #         top_k_idx = max_idx[np.argsort(max_score_np[max_idx])][::-1].tolist()

        #         for locs in top_k_idx:
        #             seq = props[locs,:]
        #             thres = mask_snip_thresh
        #             for j in thres:
        #                 filtered_seq = seq > j
        #                 integer_map = map(int,filtered_seq)
        #                 filtered_seq_int = list(integer_map)
        #                 filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                        
        #                 if 1 in filtered_seq_int:

        #                     #### getting start and end point of mask from mask branch ####

        #                     start_pt1 = filtered_seq_int2.index(1)
        #                     end_pt1 = len(filtered_seq_int2) - 1 - filtered_seq_int2[::-1].index(1) 
        #                     r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int)),operator.itemgetter(1)) if x == 1), key=len)
        #                     start_pt = r[0][0]
        #                     end_pt = r[-1][0]
        #                     if (end_pt - start_pt)/tscale > 0.02 :

        #                     #### get (start,end,cls_score,reg_score,label) for each top-k snip ####

        #                         score_ = max_score_np[locs]
        #                         score_ = 0
        #                         cls_score = score_
        #                         #lbl_id = score_map[score_]
        #                         lbl_id = 0
        #                         reg_score = np.amax(seq[start_pt+1:end_pt-1])
        #                         label = key_list[val_list.index(lbl_id)]
        #                         vid_label = key_list[val_list.index(vid_label_id)]
        #                         score_shift = np.amax(soft_cas_np[vid_label_id,start_pt:end_pt])
        #                         prop_start = start_pt1/tscale
        #                         prop_end = end_pt1/tscale
        #                         new_props.append([video_name, prop_start , prop_end , score_shift*reg_score, score_shift*cls_score,vid_label])             
        #                         for m in range(len(cas_tuple)):
        #                             start_m = cas_tuple[m][0]
        #                             end_m = cas_tuple[m][1]
        #                             score_m = cas_tuple[m][2]
        #                             reg_score = np.amax(seq[start_m:end_m])
        #                             prop_start = start_m/tscale
        #                             prop_end = end_m/tscale
        #                             cls_score = score_m
        #                             new_props.append([video_name, prop_start,prop_end,reg_score,cls_score,vid_label])
                                        
        #     ### filter duplicate proposals --> Less Time for Post-Processing #####
        #     new_props = np.stack(new_props)
        #     b_set = set(map(tuple,new_props))  
        #     result = map(list,b_set) 

        #     ### save the proposals in a csv file ###
        #     # col_name = ["video_name","xmin", "xmax", "clr_score", "reg_score","label"]
        #     # new_df = pd.DataFrame(result, columns=col_name)
        #     # new_df.to_csv("stale_output.csv", index=False)   

        #     end.record()
        #     torch.cuda.synchronize()

        #     print("Inference time :"+ str(start.elapsed_time(end)/1000) + " secs")  # milliseconds

        #     # print("Inference finished")

        del model

        
            

