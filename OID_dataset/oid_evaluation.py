#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SG_OID
@File ：evaluation.py
@Author ：jintianlei
@Date : 2023/3/6
"""
import torch
#from sgg_eval import SGNoGraphConstraintRecall, SGRecall, SGMeanRecall, SGStagewiseRecall
from utils_evaluation import evaluate_relation_of_one_image,SGRecall,SGMeanRecall
from tqdm import tqdm
import numpy as np

def eval_classic_recall(mode, groundtruths, predictions, predicate_cls_list,result_str):
    evaluator = {}
    rel_eval_result_dict = {}
    eval_recall = SGRecall(rel_eval_result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # eval_nog_recall = SGNoGraphConstraintRecall(rel_eval_result_dict)
    # eval_nog_recall.register_container(mode)
    # evaluator['eval_nog_recall'] = eval_nog_recall

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(rel_eval_result_dict, len(predicate_cls_list), predicate_cls_list,
                                    print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # eval_stagewise_recall = SGStagewiseRecall(rel_eval_result_dict)
    # eval_stagewise_recall.register_container(mode)
    # evaluator['eval_stagewise_recall'] = eval_stagewise_recall

    # prepare all inputs
    global_container = {}
    global_container['result_dict'] = rel_eval_result_dict
    global_container['mode'] = mode
    global_container['num_rel_category'] = len(predicate_cls_list)
    global_container['iou_thres'] = 0.5 #cfg.TEST.RELATION.IOU_THRESHOLD
    global_container['attribute_on'] = False

    #logger.info("evaluating relationship predictions..")
    for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(predictions)):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += "classic recall evaluations:\n"
    result_str += eval_recall.generate_print_string(mode)
    #result_str += eval_nog_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    #result_str += eval_stagewise_recall.generate_print_string(mode)

    def generate_eval_res_dict(evaluator, mode):
        res_dict = {}
        for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
            res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
        return res_dict

    # result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),
    #                                 #generate_eval_res_dict(eval_nog_recall, mode),
    #                                 generate_eval_res_dict(eval_mean_recall, mode), ])
    result_str += "\n" + "=" * 80 +"\n"

    return result_str#, result_dict_list_to_log


if __name__ =="__main__":


    mode = 'sgdet'

    eval_results = torch.load('openimage_v6_test/eval_results.pytorch', map_location=torch.device("cpu"))

    predictions = eval_results['predictions']
    groundtruths = eval_results['groundtruths']

    for _,prediction in enumerate(predictions):
        image_width,image_height = groundtruths[_].size
        # recover original size which is before transform
        predictions[_] = prediction.resize((image_width, image_height))



    predicate_cls_list = ['__background__', 'at', 'holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'on', 'ride', 'dance',
     'skateboard', 'catch', 'highfive', 'inside_of', 'eat', 'cut', 'contain', 'handshake', 'kiss', 'talk_on_phone',
     'interacts_with', 'under', 'hug', 'throw', 'hits', 'snowboard', 'kick', 'ski', 'plays', 'read']

    result_str = '\n' + '=' * 100 + '\n'

    result_str_tmp = ''
    result_str_tmp = eval_classic_recall(mode, groundtruths, predictions, predicate_cls_list, result_str_tmp)
    result_str += result_str_tmp
    print(result_str_tmp)
