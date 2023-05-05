#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：713
@File ：new_visual_predcls_val.py
@Author ：jintianlei
@Date : 2022/8/1
"""
import os
import sys
sys.path.append(os.path.abspath('.'))
from dataset import Scene_Graph_Image
import torch
import numpy as np
from vdn.VDN import visual_rel_model
from ckn.CKN import FC_Net
import json
import torchtext
from evaluation.bounding_box import BoxList
from evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, \
    SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
import cv2
import torchvision.transforms as transforms
import os
import shutil
from datetime import datetime
from datapath import image_file, test_object_label_dir, test_relation_label_dir, names, relations_names



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

    image_dir = image_file
    val_object_label = test_object_label_dir  # all .txt in train package
    val_relation_label = test_relation_label_dir

    #image_file = 'little_dataset/vg_image'
    #val_object_label = 'little_dataset/object_detection/xywh'
    #val_relation_label = 'little_dataset/relation_detection'

    filter_flag = True
    if filter_flag:
        CKN_weight = 'ckn/CKN+EPBS.pt'
        print(CKN_weight)
        CKN_model = FC_Net()
        model_dict = CKN_model.state_dict()
        pretrained_dict = torch.load(CKN_weight, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        CKN_model.load_state_dict(model_dict)
        CKN_model.to(device)


    VDN_weight = 'vdn/CKN+VDN+EPBS.pt'
    print(VDN_weight)
    VDN_model = visual_rel_model()
    model_dict = VDN_model.state_dict()
    pretrained_dict = torch.load(VDN_weight, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    VDN_model.load_state_dict(model_dict)
    VDN_model.to(device)


    word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
    word_vec_list = []
    for name_index in range(len(names)):
        word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
        word_vec_list.append(word_vec)
    word_vec_list = torch.stack(word_vec_list)

    data = Scene_Graph_Image(image_dir, val_object_label, val_relation_label, group=1, sample_num= 1, training=False)
    val_loader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0)

    predictions = []
    groundtruths = []
    iou_types = ['bbox', 'relations']
    mode = 'predcls'  # predcls sgdet
    total_count = 0
    pos_count = 0
    total_semantic_result = []
    total_fusion_result = []
    with torch.no_grad():
        for batch_i, (image_feature, batch_word_feature, object_targets, relation_target,image_name) in enumerate(val_loader):
            if (batch_i + 1) % 100 == 0:
                print(batch_i + 1, ' finished')

            image_feature = image_feature[0]#.to(device)
            object_targets = object_targets[0]#.to(device)
            relation_target = relation_target[0]#.to(device)
            batch_word_feature = batch_word_feature[0]#.to(device)
            image_name = image_name[0]#.to(device)

            predited_obj = object_targets
            obj_location_feature = torch.cat((((predited_obj[:,3]-predited_obj[:,1])/2).unsqueeze(1),((predited_obj[:,4]-predited_obj[:,2])/2).unsqueeze(1),predited_obj[:,1:]),1)
            obj_word_feature = word_vec_list[predited_obj[:, 0].long()]

            #obj = torch.cat((obj_word_feature, obj_location_feature), 1)
            obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)
            obj_id = torch.arange((len(predited_obj)))
            a_id = obj_id.repeat(len(obj_id),1).view(-1,)
            b_id = torch.repeat_interleave(obj_id,len(obj_id),dim=0)
            obj_pair = torch.cat((a_id.unsqueeze(1),b_id.unsqueeze(1)),dim=1)
            obj_pair = obj_pair[obj_pair[:,0]!=obj_pair[:,1]]

            rel_feature = torch.cat((obj[obj_pair[:, 0].long()], obj[obj_pair[:, 1].long()],
                                     obj[obj_pair[:, 1].long()][:, 50:] - obj[obj_pair[:, 0].long()][:, 50:]), 1).to(device)


            if filter_flag:
                # Use CKN to filter the results once to reduce the calculation amount of VDN
                y = CKN_model(rel_feature)
                pred_relations = torch.sigmoid(y)
                pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]
                pred_relations = pred_relations * pred_relations_conf
                pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))], device=device), pred_relations], dim=1)

                relation_max, relation_argmax = torch.max(pred_relations, dim=1)

                #filtering
                obj_pair = obj_pair[torch.sort(relation_max,descending=True)[1]][:150]
                rel_feature = rel_feature[torch.sort(relation_max,descending=True)[1]][:150]

            image = cv2.imread(image_name)
            height, width, c = image.shape
            visual_bbox_pair = torch.cat((predited_obj[obj_pair[:, 0].long()], predited_obj[obj_pair[:, 1].long()]), 1)
            visual_bbox_pair = visual_bbox_pair*torch.tensor([1,width,height,width,height,1,width,height,width,height])


            visual_bbox_pair = torch.split(visual_bbox_pair,50,dim=0)
            _rel_feature = torch.split(rel_feature,50,dim=0)

            batch_visual_pred = []
            batch_semantic_pred = []

            for j in range(len(visual_bbox_pair)):
                batch_img = []
                object_pair_location = visual_bbox_pair[j]
                current_rel_feature = _rel_feature[j]

                cut_x1, _ = torch.min(object_pair_location[:, [1, 6]], dim=1)
                cut_y1, _ = torch.min(object_pair_location[:, [2, 7]], dim=1)
                cut_x2, _ = torch.max(object_pair_location[:, [3, 8]], dim=1)
                cut_y2, _ = torch.max(object_pair_location[:, [4, 9]], dim=1)

                cut_x1 = cut_x1.int()
                cut_y1 = cut_y1.int()
                cut_x2 = cut_x2.int()
                cut_y2 = cut_y2.int()

                current_source_x1, current_source_y1, current_source_x2, current_source_y2 = object_pair_location[:,1].int(),object_pair_location[:,2].int(), object_pair_location[:,3].int(), object_pair_location[:,4].int()
                current_target_x1, current_target_y1, current_target_x2, current_target_y2 = object_pair_location[:,6].int(), object_pair_location[:,7].int(), object_pair_location[:,8].int(), object_pair_location[:,9].int()


                for i in range(len(object_pair_location)):
                    x1, y1, x2, y2 = cut_x1[i], cut_y1[i], cut_x2[i], cut_y2[i]

                    source_x1, source_y1, source_x2, source_y2 = current_source_x1[i], current_source_y1[i], current_source_x2[i], current_source_y2[i]
                    target_x1, target_y1, target_x2, target_y2 = current_target_x1[i], current_target_y1[i], current_target_x2[i], current_target_y2[i]

                    img = image[y1:y2, x1:x2]

                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                    source_mask = np.zeros((height, width), dtype=np.float32)
                    source_mask[source_y1:source_y2, source_x1:source_x2] = 1  # 1
                    target_mask = np.zeros((height, width), dtype=np.float32)
                    target_mask[target_y1:target_y2, target_x1:target_x2] = 1  # 1

                    source_mask = source_mask[y1:y2, x1:x2]
                    target_mask = target_mask[y1:y2, x1:x2]

                    source_mask = cv2.resize(source_mask, (224, 224), interpolation=cv2.INTER_AREA)
                    target_mask = cv2.resize(target_mask, (224, 224), interpolation=cv2.INTER_AREA)

                    #img = transforms.ToPILImage()(img)
                    img = transforms.ToTensor()(img)
                    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                    source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                    target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                    batch_img.append(torch.cat((source_mask, img, target_mask), 0))

                batch_img = torch.stack(batch_img).to(device)

                visual_fusion, y, fusion_relation_pred = VDN_model(batch_img, current_rel_feature)
                batch_visual_pred.append(fusion_relation_pred)
                batch_semantic_pred.append(y)

            pred_fusion_relations = torch.cat(batch_visual_pred,0)
            pred_semantic_relations = torch.cat(batch_semantic_pred, 0)


            total_semantic_result.append(pred_semantic_relations.cpu())
            total_fusion_result.append(pred_fusion_relations.cpu())

            pred_relations = pred_semantic_relations
            pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]
            pred_relations = torch.cat(
                [torch.tensor([[0] for i in range(len(pred_relations))], device=device), pred_relations], dim=1)


            pred_relations_values, pred_relation_indices = torch.sort(pred_relations, 1, descending=True)
            indices = torch.arange(0, len(pred_relations_values), device=device)
            _indices_neg = torch.repeat_interleave(indices, len(pred_relation_indices[:, 10:][0]), dim=0)
            _pred_relation_indices_neg = torch.cat(
                (_indices_neg.reshape(-1, 1), pred_relation_indices[:, 10:].reshape(-1, 1)), 1)

            _indices_pos = torch.repeat_interleave(indices, len(pred_relation_indices[:, :10][0]), dim=0)
            _pred_relation_indices_pos = torch.cat(
                (_indices_pos.reshape(-1, 1), pred_relation_indices[:, :10].reshape(-1, 1)), 1)

            pred_relations[_pred_relation_indices_neg[:, 0], _pred_relation_indices_neg[:, 1]] = 0.0

            pred_relations = pred_fusion_relations
            pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]

            pred_relations = pred_relations * pred_relations_conf
            pred_relations = torch.cat(
                [torch.tensor([[0] for i in range(len(pred_relations))], device=device), pred_relations], dim=1)
            pred_relations[_pred_relation_indices_neg[:, 0], _pred_relation_indices_neg[:, 1]] = 0.0

            relation_max, relation_argmax = torch.max(pred_relations, dim=1)
            head_semantic = torch.cat([obj_pair[:,:2].to(device),relation_argmax.view(-1, 1), relation_max.view(-1, 1), pred_relations], dim=1)
            head_semantic = head_semantic[torch.argsort(pred_relations_conf.view(1, -1), descending=True)[0]]


            boxlist = BoxList(object_targets[:, 1:5], (1, 1), 'xyxy')
            boxlist.add_field('pred_labels', object_targets[:, 0])
            boxlist.add_field('pred_scores', torch.ones(len(object_targets[:, 0])))

            if len(head_semantic) > 0:
                # print(relation_target.size(),head_semantic[:, 1:].size())
                boxlist.add_field('rel_pair_idxs', head_semantic[:, :2])  # (#rel, 2)
                boxlist.add_field('pred_rel_scores', head_semantic[:,4:])  # (#rel, #rel_class)
                boxlist.add_field('pred_rel_labels', head_semantic[:,2])  # (#rel, )
            else:
                boxlist.add_field('rel_pair_idxs', torch.tensor([[0, 0]]))  # (#rel, 2)
                boxlist.add_field('pred_rel_scores', torch.tensor([[0 for i in range(51)]]))  # (#rel, #rel_class)
                boxlist.add_field('pred_rel_labels', torch.tensor([0]))  # (#rel, )

            _boxlist = BoxList(object_targets[:, 1:5], (1, 1), 'xyxy')
            _boxlist.add_field('labels', object_targets[:, 0])
            _boxlist.add_field('relation_tuple', relation_target[:, :])

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

