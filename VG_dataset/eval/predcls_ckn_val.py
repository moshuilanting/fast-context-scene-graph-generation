#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：713
@File ：new_predcls_val.py
@Author ：jintianlei
@Date : 2022/7/29
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from dataset import Scene_Graph
import torch
import numpy as np
from ckn.CKN import FC_Net
import json
from evaluation.bounding_box import BoxList
from evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
import torchtext
from datapath import test_object_label_dir, test_relation_label_dir, names, relations_names


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_scores').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # No Graph Constraint Zero-Shot Recall
    evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

    return

if __name__ == "__main__":
    device = torch.device('cuda', 0)

    model = FC_Net()
    model_dict = model.state_dict()
    # pretrained_dict = torch.load('runs/0708_EPBS.pt')
    # pretrained_dict = {'image_conv.' + k: v for k, v in pretrained_dict['model'].items() if
    #                    'image_conv.' + k in model_dict}
    pretrained_dict = torch.load('ckn/CKN+EPBS.pt', map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    val_object_label_dir = test_object_label_dir  # all .txt in train package
    val_relation_label_dir = test_relation_label_dir


    word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
    word_vec_list = []
    for name_index in range(len(names)):
        word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
        word_vec_list.append(word_vec)
    word_vec_list = torch.stack(word_vec_list).to(device)

    data = Scene_Graph(val_object_label_dir, val_relation_label_dir, training = False)
    val_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=1, #batch_size ==1
                                               shuffle=False,
                                               num_workers=1)


    predictions = []
    groundtruths = []
    iou_types = ['bbox', 'relations']
    mode = 'predcls' #predcls sgdet


    with torch.no_grad():
        for batch_i, (relations_feature,relations,object_targets,relation_targets) in enumerate(val_loader):
            if (batch_i+1) % 1000 == 0:
                print(batch_i+1,' finished')

            object_targets = object_targets[0].to(device)
            relation_targets = relation_targets[0].to(device)
            predited_obj = object_targets

            obj_location_feature = torch.cat((((predited_obj[:,3]-predited_obj[:,1])/2).unsqueeze(1),((predited_obj[:,4]-predited_obj[:,2])/2).unsqueeze(1),predited_obj[:,1:]),1)
            obj_word_feature = word_vec_list[predited_obj[:, 0].long()]
            #print(obj_word_feature.size(),obj_location_feature.size())            #obj = torch.cat((obj_word_feature, obj_location_feature), 1)
            obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)
            obj_id = torch.arange((len(predited_obj)))
            a_id = obj_id.repeat(len(obj_id),1).view(-1,)
            b_id = torch.repeat_interleave(obj_id,len(obj_id),dim=0)
            obj_pair = torch.cat((a_id.unsqueeze(1),b_id.unsqueeze(1)),dim=1)
            obj_pair = obj_pair[obj_pair[:,0]!=obj_pair[:,1]]

            #[word_vector,c_x,c_y,x1,y1,x2,y2 ,word_vector,c_x,c_y,x1,y1,x2,y2, dc_x, dc_y, dx1,dx2,dy2,dy2]
            rel_feature = torch.cat((obj[obj_pair[:, 0].long()], obj[obj_pair[:, 1].long()], obj[obj_pair[:, 1].long()][:, 50:] - obj[obj_pair[:, 0].long()][:, 50:]),1)

            relations = relations.to(device)
            y = model(rel_feature)
            pred_relations = torch.sigmoid(y)

            pred_relations_conf, pred_relations = pred_relations[:,0:1],pred_relations[:,1:]
            pred_relations = pred_relations*pred_relations_conf

            pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))], device=device), pred_relations], dim=1)

            relation_max, relation_argmax = torch.max(pred_relations, dim=1)

            head_semantic = torch.cat([obj_pair[:,:2].to(device),relation_argmax.view(-1, 1), relation_max.view(-1, 1), pred_relations], dim=1)
            head_semantic = head_semantic[torch.argsort(pred_relations_conf.view(1,-1), descending=True)[0]]
            #head_semantic = head_semantic[torch.argsort(head_semantic[:, 3], descending=True)]



            boxlist = BoxList(object_targets[:, 1:5], (1, 1), 'xyxy')
            boxlist.add_field('pred_labels', object_targets[:, 0])
            boxlist.add_field('pred_scores', torch.ones(len(object_targets[:, 0])))

            if len(head_semantic) > 0:
                #print(relation_target.size(),head_semantic[:, 1:].size())
                boxlist.add_field('rel_pair_idxs',head_semantic[:, :2])  # (#rel, 2)
                boxlist.add_field('pred_rel_scores', head_semantic[:,4:])  # (#rel, #rel_class)
                boxlist.add_field('pred_rel_labels', head_semantic[:,2])  # (#rel, )
            else:
                boxlist.add_field('rel_pair_idxs', torch.tensor([[0,0]]))  # (#rel, 2)
                boxlist.add_field('pred_rel_scores', torch.tensor([[0 for i in range(51)]]))  # (#rel, #rel_class)
                boxlist.add_field('pred_rel_labels', torch.tensor([0]))  # (#rel, )
            #print(head_semantic[:, :3])

            _boxlist = BoxList(object_targets[:, 1:5], (1, 1), 'xyxy')
            _boxlist.add_field('labels', object_targets[:, 0])
            _boxlist.add_field('relation_tuple', relation_targets[:, :])

            predictions.append(boxlist)
            groundtruths.append(_boxlist)


    torch.save({'predictions': predictions}, 'test.pytorch')
    torch.save({'groundtruths': groundtruths}, 'label.pytorch')

    predictions = torch.load('test.pytorch', map_location=torch.device("cpu"))['predictions']
    groundtruths = torch.load('label.pytorch', map_location=torch.device("cpu"))['groundtruths']

    attribute_on = False
    num_attributes = 201
    num_rel_category = 51
    multiple_preds = False
    iou_thres = 0.45

    zeroshot_triplet = torch.load("evaluation/vg/zeroshot_triplet.pytorch",
                                  map_location=torch.device("cpu")).long().numpy()

    result_str = '\n' + '=' * 100 + '\n'
    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(result_dict)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        # test on no graph constraint zero-shot recall
        eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, relations_names, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, relations_names,
                                             print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes

        for groundtruth, prediction in zip(groundtruths, predictions):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)

        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_zeroshot_recall.generate_print_string(mode)
        result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)

        USE_GT_BOX = True
        if USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    print(result_str)
