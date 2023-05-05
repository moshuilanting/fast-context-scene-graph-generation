#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PSG_TEST
@File ：sgg_eval.py
@Author ：jintianlei
@Date : 2023/3/8
"""

# ---------------------------------------------------------------
# vg_eval.py
# Set-up time: 2020/5/18 上午9:48
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import json
import os

import numpy as np
import torch


import itertools
from abc import ABC, abstractmethod
from functools import reduce
from terminaltables import AsciiTable

from PIL import Image
from tqdm import tqdm
from psg_infer import rgb2id


class SceneGraphEvaluation(ABC):
    def __init__(self,
                 result_dict,
                 nogc_result_dict,
                 nogc_thres_num,
                 detection_method='bbox'):
        super().__init__()
        self.result_dict = result_dict
        self.nogc_result_dict = nogc_result_dict
        self.nogc_thres_num = nogc_thres_num

        self.detection_method = detection_method
        if detection_method not in ('bbox', 'pan_seg'):
            print('invalid detection method. using bbox instead.')
            self.detection_method = detection_method = 'bbox'

        if detection_method == 'bbox':
            self.generate_triplet = _triplet_bbox
            self.compute_pred_matches = _compute_pred_matches_bbox
        elif detection_method == 'pan_seg':
            self.generate_triplet = _triplet_panseg
            self.compute_pred_matches = _compute_pred_matches_panseg

    @abstractmethod
    def register_container(self, mode):
        print('Register Result Container')
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print('Generate Print String')
        pass


class SGRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGRecall, self).__init__(*args, **kwargs)

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}
        self.nogc_result_dict[mode + '_recall'] = {
            ngc: {
                20: [],
                50: [],
                100: []
            }
            for ngc in self.nogc_thres_num
        }
        if mode == 'sgdet':
            self.result_dict['phrdet_recall'] = {20: [], 50: [], 100: []}
            self.nogc_result_dict['phrdet_recall'] = {
                ngc: {
                    20: [],
                    50: [],
                    100: []
                }
                for ngc in self.nogc_thres_num
            }

    def _calculate_single(self,
                          target_dict,
                          prediction_to_gt,
                          gt_rels,
                          mode,
                          nogc_num=None):
        target = target_dict[mode +
                             '_recall'] if nogc_num is None else target_dict[
                                 mode + '_recall'][nogc_num]
        for k in target:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, prediction_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            target[k].append(rec_i)

    def _print_single(self, target_dict, mode, nogc_num=None):
        target = target_dict[mode +
                             '_recall'] if nogc_num is None else target_dict[
                                 mode + '_recall'][nogc_num]
        result_str = 'SGG eval: '
        for k, v in target.items():
            result_str += ' R @ %d: %.4f; ' % (k, np.mean(v))
        suffix_type = 'Recall.' if nogc_num is None else 'NoGraphConstraint @ %d Recall.' % nogc_num
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'
        return result_str

    def generate_print_string(self, mode):
        result_str = self._print_single(self.result_dict, mode)
        if mode == 'sgdet':
            result_str += self._print_single(self.result_dict, 'phrdet')

        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict, mode,
                                             nogc_num)
            if mode == 'sgdet':
                result_str += self._print_single(self.nogc_result_dict,
                                                 'phrdet', nogc_num)

        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        pred_masks = local_container['pred_masks']
        gt_masks = local_container['gt_masks']
        if mode == 'predcls':
            pred_masks = gt_masks

        iou_thrs = global_container['iou_thrs']

        nogc_thres_num = self.nogc_thres_num

        if self.detection_method == 'bbox':
            gt_det_results = gt_boxes
        if self.detection_method == 'pan_seg':
            gt_det_results = gt_masks
        gt_triplets, gt_triplet_det_results, _ = self.generate_triplet(
            gt_rels, gt_classes, gt_det_results)

        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_det_results'] = gt_triplet_det_results
        # if self.detection_method == 'bbox':
        #     local_container['gt_triplet_boxes'] = gt_triplet_det_results
        # if self.detection_method == 'pan_seg':
        #     local_container['gt_triplet_masks'] = gt_triplet_det_results

        # compute the graph constraint setting pred_rels
        pred_rels = np.column_stack(
            (pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)

        if self.detection_method == 'bbox':
            pred_det_results = pred_boxes
        if self.detection_method == 'pan_seg':
            pred_det_results = pred_masks
        pred_triplets, pred_triplet_det_results, _ = \
            self.generate_triplet(
            pred_rels, pred_classes, pred_det_results, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        # if mode is sgdet, report both sgdet and phrdet

        pred_to_gt = self.compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_det_results,
            pred_triplet_det_results,
            iou_thrs,
            phrdet=False,
        )

        local_container['pred_to_gt'] = pred_to_gt
        self._calculate_single(self.result_dict, pred_to_gt, gt_rels, mode)

        if mode == 'sgdet':
            pred_to_gt = self.compute_pred_matches(gt_triplets,
                                                   pred_triplets,
                                                   gt_triplet_det_results,
                                                   pred_triplet_det_results,
                                                   iou_thrs,
                                                   phrdet=True)
            local_container['phrdet_pred_to_gt'] = pred_to_gt
            self._calculate_single(self.result_dict,
                                   pred_to_gt,
                                   gt_rels,
                                   mode='phrdet')

        if self.detection_method != 'pan_seg':
            # compute the no graph constraint setting pred_rels
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:,
                                                                           1:]
            sorted_inds = np.argsort(nogc_overall_scores, axis=-1)[:, ::-1]
            sorted_nogc_overall_scores = np.sort(nogc_overall_scores,
                                                 axis=-1)[:, ::-1]
            gt_pair_idx = gt_rels[:, 0] * 10000 + gt_rels[:, 1]
            for nogc_num in nogc_thres_num:
                nogc_score_inds_ = argsort_desc(
                    sorted_nogc_overall_scores[:, :nogc_num])
                nogc_pred_rels = np.column_stack(
                    (pred_rel_inds[nogc_score_inds_[:, 0]],
                     sorted_inds[nogc_score_inds_[:, 0],
                                 nogc_score_inds_[:, 1]] + 1))
                nogc_pred_scores = rel_scores[
                    nogc_score_inds_[:, 0],
                    sorted_inds[nogc_score_inds_[:, 0],
                                nogc_score_inds_[:, 1]] + 1]

                pred_triplets, pred_triplet_det_results, pred_triplet_scores =\
                    self.generate_triplet(
                    nogc_pred_rels, pred_classes, pred_det_results,
                    nogc_pred_scores, obj_scores)

                # prepare the gt rel signal to be used in PairAccuracy:
                pred_pair_idx = nogc_pred_rels[:,
                                               0] * 10000 + nogc_pred_rels[:,
                                                                           1]
                local_container['nogc@%d_pred_pair_in_gt' % nogc_num] = \
                    (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

                # Compute recall. It's most efficient to match once and then do recall after
                pred_to_gt = self.compute_pred_matches(
                    gt_triplets,
                    pred_triplets,
                    gt_triplet_det_results,
                    pred_triplet_det_results,
                    iou_thrs,
                    phrdet=False,
                )
                # NOTE: For NGC recall, zs recall, mean recall, only need to crop the top 100 triplets.
                # While for computing the Pair Accuracy, all of the pairs are needed here.
                local_container['nogc@%d_pred_to_gt' %
                                nogc_num] = pred_to_gt[:100]  # for zR, mR, R
                local_container['nogc@%d_all_pred_to_gt' %
                                nogc_num] = pred_to_gt  # for Pair accuracy
                self._calculate_single(self.nogc_result_dict, pred_to_gt[:100],
                                       gt_rels, mode, nogc_num)

                if mode == 'sgdet':
                    pred_to_gt = self.compute_pred_matches(
                        gt_triplets,
                        pred_triplets,
                        gt_triplet_det_results,
                        pred_triplet_det_results,
                        iou_thrs,
                        phrdet=True,
                    )
                    local_container['phrdet_nogc@%d_pred_to_gt' %
                                    nogc_num] = pred_to_gt[:100]
                    local_container['phrdet_nogc@%d_all_pred_to_gt' %
                                    nogc_num] = pred_to_gt  # for Pair accuracy
                    self._calculate_single(self.nogc_result_dict,
                                           pred_to_gt[:100],
                                           gt_rels,
                                           mode='phrdet',
                                           nogc_num=nogc_num)

        return local_container


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self,
                 result_dict,
                 nogc_result_dict,
                 nogc_thres_num,
                 num_rel,
                 ind_to_predicates,
                 detection_method='pan_seg',
                 print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict, nogc_result_dict,
                                           nogc_thres_num, detection_method)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {
            20: [[] for _ in range(self.num_rel)],
            50: [[] for _ in range(self.num_rel)],
            100: [[] for _ in range(self.num_rel)]
        }
        self.result_dict[mode + '_mean_recall_list'] = {
            20: [],
            50: [],
            100: []
        }

        self.nogc_result_dict[mode + '_mean_recall'] = {
            ngc: {
                20: 0.0,
                50: 0.0,
                100: 0.0
            }
            for ngc in self.nogc_thres_num
        }
        self.nogc_result_dict[mode + '_mean_recall_collect'] = {
            ngc: {
                20: [[] for _ in range(self.num_rel)],
                50: [[] for _ in range(self.num_rel)],
                100: [[] for _ in range(self.num_rel)]
            }
            for ngc in self.nogc_thres_num
        }
        self.nogc_result_dict[mode + '_mean_recall_list'] = {
            ngc: {
                20: [],
                50: [],
                100: []
            }
            for ngc in self.nogc_thres_num
        }

        if mode == 'sgdet':
            self.result_dict['phrdet_mean_recall'] = {
                20: 0.0,
                50: 0.0,
                100: 0.0
            }
            self.result_dict['phrdet_mean_recall_collect'] = {
                20: [[] for _ in range(self.num_rel)],
                50: [[] for _ in range(self.num_rel)],
                100: [[] for _ in range(self.num_rel)]
            }
            self.result_dict['phrdet_mean_recall_list'] = {
                20: [],
                50: [],
                100: []
            }

            self.nogc_result_dict['phrdet_mean_recall'] = {
                ngc: {
                    20: 0.0,
                    50: 0.0,
                    100: 0.0
                }
                for ngc in self.nogc_thres_num
            }
            self.nogc_result_dict['phrdet_mean_recall_collect'] = {
                ngc: {
                    20: [[] for _ in range(self.num_rel)],
                    50: [[] for _ in range(self.num_rel)],
                    100: [[] for _ in range(self.num_rel)]
                }
                for ngc in self.nogc_thres_num
            }
            self.nogc_result_dict['phrdet_mean_recall_list'] = {
                ngc: {
                    20: [],
                    50: [],
                    100: []
                }
                for ngc in self.nogc_thres_num
            }

    def _collect_single(self,
                        target_dict,
                        prediction_to_gt,
                        gt_rels,
                        mode,
                        nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]

        for k in target_collect:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, prediction_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    target_collect[k][n].append(
                        float(recall_hit[n] / recall_count[n]))

    def _calculate_single(self, target_dict, mode, nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]
        target_recall = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]
        for k, v in target_recall.items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(target_collect[k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(target_collect[k][idx + 1])
                target_recall_list[k].append(tmp_recall)
                sum_recall += tmp_recall
            target_recall[k] = sum_recall / float(num_rel_no_bg)

    def _print_single(self,
                      target_dict,
                      mode,
                      nogc_num=None,
                      predicate_freq=None):
        target = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]

        result_str = 'SGG eval: '
        for k, v in target.items():
            result_str += ' mR @ %d: %.4f; ' % (k, float(v))
        suffix_type = 'Mean Recall.' if nogc_num is None else 'NoGraphConstraint @ %d Mean Recall.' % (
            nogc_num)
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'

        # result_str is flattened for copying the data to the form, while the table is for vis.
        # Only for graph constraint, one mode for short
        if self.print_detail and mode != 'phrdet' and nogc_num is None:
            rel_name_list, res = self.rel_name_list, target_recall_list[100]
            if predicate_freq is not None:
                rel_name_list = [
                    self.rel_name_list[sid] for sid in predicate_freq
                ]
                res = [target_recall_list[100][sid] for sid in predicate_freq]

            result_per_predicate = []
            for n, r in zip(rel_name_list, res):
                result_per_predicate.append(
                    ('{}'.format(str(n)), '{:.4f}'.format(r)))
            result_str += '\t'.join(list(map(str, rel_name_list)))
            result_str += '\n'

            def map_float(num):
                return '{:.4f}'.format(num)

            result_str += '\t'.join(list(map(map_float, res)))
            result_str += '\n'

            num_columns = min(6, len(result_per_predicate) * 2)
            results_flatten = list(itertools.chain(*result_per_predicate))
            headers = ['predicate', 'Rec100'] * (num_columns // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            result_str += table.table + '\n'

        return result_str

    def generate_print_string(self, mode, predicate_freq=None):
        result_str = self._print_single(self.result_dict,
                                        mode,
                                        predicate_freq=predicate_freq)
        if mode == 'sgdet':
            result_str += self._print_single(self.result_dict,
                                             'phrdet',
                                             predicate_freq=predicate_freq)

        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict,
                                             mode,
                                             nogc_num,
                                             predicate_freq=predicate_freq)
            if mode == 'sgdet':
                result_str += self._print_single(self.nogc_result_dict,
                                                 'phrdet',
                                                 nogc_num,
                                                 predicate_freq=predicate_freq)
        return result_str

    def collect_mean_recall_items(self, global_container, local_container,
                                  mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        self._collect_single(self.result_dict, pred_to_gt, gt_rels, mode)
        if mode == 'sgdet':
            phrdet_pred_to_gt = local_container['phrdet_pred_to_gt']
            self._collect_single(self.result_dict, phrdet_pred_to_gt, gt_rels,
                                 'phrdet')

        if self.detection_method != 'pan_seg':
            for nogc_num in self.nogc_thres_num:
                nogc_pred_to_gt = local_container['nogc@%d_pred_to_gt' %
                                                  nogc_num]
                self._collect_single(self.nogc_result_dict, nogc_pred_to_gt,
                                     gt_rels, mode, nogc_num)

                if mode == 'sgdet':
                    nogc_pred_to_gt = local_container[
                        'phrdet_nogc@%d_pred_to_gt' % nogc_num]
                    self._collect_single(self.nogc_result_dict,
                                         nogc_pred_to_gt, gt_rels, 'phrdet',
                                         nogc_num)

    def calculate_mean_recall(self, mode):
        self._calculate_single(self.result_dict, mode)
        if mode == 'sgdet':
            self._calculate_single(self.result_dict, 'phrdet')

        for nogc_num in self.nogc_thres_num:
            self._calculate_single(self.nogc_result_dict, mode, nogc_num)
            if mode == 'sgdet':
                self._calculate_single(self.nogc_result_dict, 'phrdet',
                                       nogc_num)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""


class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGPairAccuracy, self).__init__(*args, **kwargs)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.nogc_result_dict[mode + '_accuracy_hit'] = {
            ngc: {
                20: [],
                50: [],
                100: []
            }
            for ngc in self.nogc_thres_num
        }
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}
        self.nogc_result_dict[mode + '_accuracy_count'] = {
            ngc: {
                20: [],
                50: [],
                100: []
            }
            for ngc in self.nogc_thres_num
        }

    def _calculate_single(self,
                          target_dict,
                          prediction_to_gt,
                          gt_rels,
                          mode,
                          pred_pair_in_gt,
                          nogc_num=None):
        target_hit = target_dict[mode + '_accuracy_hit'] if nogc_num is None else \
            target_dict[mode + '_accuracy_hit'][nogc_num]
        target_count = target_dict[mode + '_accuracy_count'] if nogc_num is None else \
            target_dict[mode + '_accuracy_count'][nogc_num]

        if mode != 'sgdet':
            gt_pair_pred_to_gt = []
            for p, flag in zip(prediction_to_gt, pred_pair_in_gt):
                if flag:
                    gt_pair_pred_to_gt.append(p)
            for k in target_hit:
                # to calculate accuracy, only consider those gt pairs
                # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                target_hit[k].append(float(len(gt_pair_match)))
                target_count[k].append(float(gt_rels.shape[0]))

    def _print_single(self, target_dict, mode, nogc_num=None):
        target_hit = target_dict[mode + '_accuracy_hit'] if nogc_num is None else \
            target_dict[mode + '_accuracy_hit'][nogc_num]
        target_count = target_dict[mode + '_accuracy_count'] if nogc_num is None else \
            target_dict[mode + '_accuracy_count'][nogc_num]

        result_str = 'SGG eval: '
        for k, v in target_hit.items():
            a_hit = np.mean(v)
            a_count = np.mean(target_count[k])
            result_str += ' A @ %d: %.4f; ' % (k, a_hit / a_count)
        suffix_type = 'TopK Accuracy.' if nogc_num is None else 'NoGraphConstraint @ %d TopK Accuracy.' % (
            nogc_num)
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'
        return result_str

    def generate_print_string(self, mode):
        result_str = self._print_single(self.result_dict, mode)
        if mode == 'sgdet':
            result_str += self._print_single(self.result_dict, 'phrdet')

        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict, mode,
                                             nogc_num)
            if mode == 'sgdet':
                result_str += self._print_single(self.nogc_result_dict,
                                                 'phrdet', nogc_num)
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container[
            'pred_rel_inds'][:,
                             0] * 10000 + local_container['pred_rel_inds'][:,
                                                                           1]
        gt_pair_idx = local_container[
            'gt_rels'][:, 0] * 10000 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None]
                                == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        if mode != 'sgdet':
            pred_to_gt = local_container['pred_to_gt']
            gt_rels = local_container['gt_rels']

            self._calculate_single(self.result_dict, pred_to_gt, gt_rels, mode,
                                   self.pred_pair_in_gt)

            if self.detection_method != 'pan_seg':
                # nogc
                for nogc_num in self.nogc_thres_num:

                    nogc_pred_to_gt = local_container['nogc@%d_all_pred_to_gt'
                                                      % nogc_num]
                    self._calculate_single(
                        self.nogc_result_dict, nogc_pred_to_gt, gt_rels, mode,
                        local_container['nogc@%d_pred_pair_in_gt' % nogc_num],
                        nogc_num)


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self,
                 result_dict,
                 nogc_result_dict,
                 nogc_thres_num,
                 num_rel,
                 ind_to_predicates,
                 detection_method='pan_seg',
                 print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict, nogc_result_dict,
                                           nogc_thres_num, detection_method)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {
            20: [[] for _ in range(self.num_rel)],
            50: [[] for _ in range(self.num_rel)],
            100: [[] for _ in range(self.num_rel)]
        }
        self.result_dict[mode + '_mean_recall_list'] = {
            20: [],
            50: [],
            100: []
        }

        self.nogc_result_dict[mode + '_mean_recall'] = {
            ngc: {
                20: 0.0,
                50: 0.0,
                100: 0.0
            }
            for ngc in self.nogc_thres_num
        }
        self.nogc_result_dict[mode + '_mean_recall_collect'] = {
            ngc: {
                20: [[] for _ in range(self.num_rel)],
                50: [[] for _ in range(self.num_rel)],
                100: [[] for _ in range(self.num_rel)]
            }
            for ngc in self.nogc_thres_num
        }
        self.nogc_result_dict[mode + '_mean_recall_list'] = {
            ngc: {
                20: [],
                50: [],
                100: []
            }
            for ngc in self.nogc_thres_num
        }

        if mode == 'sgdet':
            self.result_dict['phrdet_mean_recall'] = {
                20: 0.0,
                50: 0.0,
                100: 0.0
            }
            self.result_dict['phrdet_mean_recall_collect'] = {
                20: [[] for _ in range(self.num_rel)],
                50: [[] for _ in range(self.num_rel)],
                100: [[] for _ in range(self.num_rel)]
            }
            self.result_dict['phrdet_mean_recall_list'] = {
                20: [],
                50: [],
                100: []
            }

            self.nogc_result_dict['phrdet_mean_recall'] = {
                ngc: {
                    20: 0.0,
                    50: 0.0,
                    100: 0.0
                }
                for ngc in self.nogc_thres_num
            }
            self.nogc_result_dict['phrdet_mean_recall_collect'] = {
                ngc: {
                    20: [[] for _ in range(self.num_rel)],
                    50: [[] for _ in range(self.num_rel)],
                    100: [[] for _ in range(self.num_rel)]
                }
                for ngc in self.nogc_thres_num
            }
            self.nogc_result_dict['phrdet_mean_recall_list'] = {
                ngc: {
                    20: [],
                    50: [],
                    100: []
                }
                for ngc in self.nogc_thres_num
            }

    def _collect_single(self,
                        target_dict,
                        prediction_to_gt,
                        gt_rels,
                        mode,
                        nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]

        for k in target_collect:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, prediction_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    target_collect[k][n].append(
                        float(recall_hit[n] / recall_count[n]))

    def _calculate_single(self, target_dict, mode, nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]
        target_recall = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]
        for k, v in target_recall.items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(target_collect[k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(target_collect[k][idx + 1])
                target_recall_list[k].append(tmp_recall)
                sum_recall += tmp_recall
            target_recall[k] = sum_recall / float(num_rel_no_bg)

    def _print_single(self,
                      target_dict,
                      mode,
                      nogc_num=None,
                      predicate_freq=None):
        target = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]

        result_str = 'SGG eval: '
        for k, v in target.items():
            result_str += ' mR @ %d: %.4f; ' % (k, float(v))
        suffix_type = 'Mean Recall.' if nogc_num is None else 'NoGraphConstraint @ %d Mean Recall.' % (
            nogc_num)
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'

        # result_str is flattened for copying the data to the form, while the table is for vis.
        # Only for graph constraint, one mode for short
        if self.print_detail and mode != 'phrdet' and nogc_num is None:
            rel_name_list, res = self.rel_name_list, target_recall_list[100]
            if predicate_freq is not None:
                rel_name_list = [
                    self.rel_name_list[sid] for sid in predicate_freq
                ]
                res = [target_recall_list[100][sid] for sid in predicate_freq]

            result_per_predicate = []
            for n, r in zip(rel_name_list, res):
                result_per_predicate.append(
                    ('{}'.format(str(n)), '{:.4f}'.format(r)))
            result_str += '\t'.join(list(map(str, rel_name_list)))
            result_str += '\n'

            def map_float(num):
                return '{:.4f}'.format(num)

            result_str += '\t'.join(list(map(map_float, res)))
            result_str += '\n'

            num_columns = min(6, len(result_per_predicate) * 2)
            results_flatten = list(itertools.chain(*result_per_predicate))
            headers = ['predicate', 'Rec100'] * (num_columns // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            result_str += table.table + '\n'

        return result_str

    def generate_print_string(self, mode, predicate_freq=None):
        result_str = self._print_single(self.result_dict,
                                        mode,
                                        predicate_freq=predicate_freq)
        if mode == 'sgdet':
            result_str += self._print_single(self.result_dict,
                                             'phrdet',
                                             predicate_freq=predicate_freq)

        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict,
                                             mode,
                                             nogc_num,
                                             predicate_freq=predicate_freq)
            if mode == 'sgdet':
                result_str += self._print_single(self.nogc_result_dict,
                                                 'phrdet',
                                                 nogc_num,
                                                 predicate_freq=predicate_freq)
        return result_str

    def collect_mean_recall_items(self, global_container, local_container,
                                  mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        self._collect_single(self.result_dict, pred_to_gt, gt_rels, mode)
        if mode == 'sgdet':
            phrdet_pred_to_gt = local_container['phrdet_pred_to_gt']
            self._collect_single(self.result_dict, phrdet_pred_to_gt, gt_rels,
                                 'phrdet')

        if self.detection_method != 'pan_seg':
            for nogc_num in self.nogc_thres_num:
                nogc_pred_to_gt = local_container['nogc@%d_pred_to_gt' %
                                                  nogc_num]
                self._collect_single(self.nogc_result_dict, nogc_pred_to_gt,
                                     gt_rels, mode, nogc_num)

                if mode == 'sgdet':
                    nogc_pred_to_gt = local_container[
                        'phrdet_nogc@%d_pred_to_gt' % nogc_num]
                    self._collect_single(self.nogc_result_dict,
                                         nogc_pred_to_gt, gt_rels, 'phrdet',
                                         nogc_num)

    def calculate_mean_recall(self, mode):
        self._calculate_single(self.result_dict, mode)
        if mode == 'sgdet':
            self._calculate_single(self.result_dict, 'phrdet')

        for nogc_num in self.nogc_thres_num:
            self._calculate_single(self.nogc_result_dict, mode, nogc_num)
            if mode == 'sgdet':
                self._calculate_single(self.nogc_result_dict, 'phrdet',
                                       nogc_num)






def sgg_evaluation(
    mode,
    groundtruths,
    predictions,
    iou_thrs,
    logger,
    ind_to_predicates,
    multiple_preds=False,
    predicate_freq=None,
    nogc_thres_num=None,
    detection_method='bbox',
):
    modes = mode if isinstance(mode, list) else [mode]
    result_container = dict()
    for m in modes:
        msg = 'Evaluating {}...'.format(m)
        if logger is None:
            msg = '\n' + msg
        #print_log(msg, logger=logger)
        print(msg)
        single_result_dict = vg_evaluation_single(
            m,
            groundtruths,
            predictions,
            iou_thrs,
            logger,
            ind_to_predicates,
            multiple_preds,
            predicate_freq,
            nogc_thres_num,
            detection_method,
        )
        result_container.update(single_result_dict)
    return result_container


def vg_evaluation_single(
    mode,
    groundtruths,
    predictions,
    iou_thrs,
    logger,
    ind_to_predicates,
    multiple_preds=False,
    predicate_freq=None,
    nogc_thres_num=None,
    detection_method='bbox',
):
    # # get zeroshot triplet
    num_predicates = len(ind_to_predicates)

    assert isinstance(nogc_thres_num,
                      (list, tuple, int)) or nogc_thres_num is None
    if nogc_thres_num is None:
        nogc_thres_num = [num_predicates - 1]  # default: all
    elif isinstance(nogc_thres_num, int):
        nogc_thres_num = [nogc_thres_num]
    else:
        pass

    result_str = '\n' + '=' * 100 + '\n'
    result_dict = {}
    nogc_result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict,
                           nogc_result_dict,
                           nogc_thres_num,
                           detection_method=detection_method)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
    eval_pair_accuracy = SGPairAccuracy(result_dict, nogc_result_dict,
                                        nogc_thres_num, detection_method)
    eval_pair_accuracy.register_container(mode)
    evaluator['eval_pair_accuracy'] = eval_pair_accuracy

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(
        result_dict,
        nogc_result_dict,
        nogc_thres_num,
        num_predicates,
        ind_to_predicates,
        detection_method=detection_method,
        print_detail=True,
    )
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # prepare all inputs
    global_container = {}
    # global_container["zeroshot_triplet"] = zeroshot_triplet
    global_container['result_dict'] = result_dict
    global_container['mode'] = mode
    global_container['multiple_preds'] = multiple_preds
    global_container['num_predicates'] = num_predicates
    global_container['iou_thrs'] = iou_thrs
    # global_container['attribute_on'] = attribute_on
    # global_container['num_attributes'] = num_attributes

    #pbar = mmcv.ProgressBar(len(groundtruths))
    for groundtruth, prediction in zip(groundtruths, predictions):
        # Skip empty predictions
        if prediction.refine_bboxes is None:
            continue

        evaluate_relation_of_one_image(groundtruth, prediction,
                                       global_container, evaluator)
        #pbar.update()

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode, predicate_freq)
    if mode != 'sgdet':
        result_str += eval_pair_accuracy.generate_print_string(mode)
    result_str += '=' * 100 + '\n'

    if logger is None:
        result_str = '\n' + result_str
    print(result_str)
    #print_log(result_str, logger=logger)

    return format_result_dict(result_dict, result_str, mode)


def format_result_dict(result_dict, result_str, mode):
    """
    Function:
        This is used for getting the results in both float data form and text
        form so that they can be logged into tensorboard (scalar and text).

        Here we only log the graph constraint results excluding phrdet.
    """
    formatted = dict()
    copy_stat_str = ''
    # Traditional Recall
    for k, v in result_dict[mode + '_recall'].items():
        formatted[mode + '_recall_' + 'R_%d' % k] = np.mean(v)
        copy_stat_str += (mode + '_recall_' + 'R_%d: ' % k +
                          '{:0.3f}'.format(np.mean(v)) + '\n')
    # mean recall
    for k, v in result_dict[mode + '_mean_recall'].items():
        formatted[mode + '_mean_recall_' + 'mR_%d' % k] = float(v)
        copy_stat_str += (mode + '_mean_recall_' + 'mR_%d: ' % k +
                          '{:0.3f}'.format(float(v)) + '\n')
    if mode != 'sgdet':
        # Accuracy
        for k, v in result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(result_dict[mode + '_accuracy_count'][k])
            formatted[mode + '_accuracy_hit_' + 'A_%d' % k] = a_hit / a_count
            copy_stat_str += (mode + '_accuracy_hit_' + 'A_%d: ' % k +
                              '{:0.3f}'.format(a_hit / a_count) + '\n')
    formatted[mode + '_copystat'] = copy_stat_str

    formatted[mode + '_runtime_eval_str'] = result_str
    return formatted


def evaluate_relation_of_one_image(groundtruth, prediction, global_container,
                                   evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.rels

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.bboxes  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.labels  # (#gt_objs, )

    # about relations
    local_container[
        'pred_rel_inds'] = prediction.rel_pair_idxes  # (#pred_rels, 2)
    local_container[
        'rel_scores'] = prediction.rel_dists  # (#pred_rels, num_pred_class)

    # about objects
    local_container[
        'pred_boxes'] = prediction.refine_bboxes[:, :4]  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.labels  # (#pred_objs, )
    local_container[
        'obj_scores'] = prediction.refine_bboxes[:, -1]  # (#pred_objs, )

    # about pan_seg masks
    local_container['gt_masks'] = groundtruth.masks
    local_container['pred_masks'] = prediction.masks

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph
    # Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    # evaluator["eval_zeroshot_recall"].prepare_zeroshot(
    #     global_container, local_container
    # )

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(
            local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if (local_container['gt_boxes'].shape[0] !=
                local_container['pred_boxes'].shape[0]):
            print(
                'Num of GT boxes is not matching with num of pred boxes in SGCLS'
            )
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
        rel_scores_sorted = np.column_stack(
            (pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first
    # (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(
        global_container, local_container, mode)

    # No Graph Constraint
    # evaluator['eval_nog_recall'].calculate_recall(global_container,
    # local_container, mode)

    # Zero shot Recall
    # evaluator["eval_zeroshot_recall"].calculate_recall(
    #     global_container, local_container, mode
    # )

    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container,
                                                     local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(
        global_container, local_container, mode)

    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """from list of attribute indexes to [1,0,1,0,...,0,1] form."""
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes),
                                    device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets


def _triplet_bbox(relations,
                  classes,
                  boxes,
                  predicate_scores=None,
                  class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:,
                                                                            2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id],
            predicate_scores,
            class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches_bbox(gt_triplets,
                               pred_triplets,
                               gt_boxes,
                               pred_boxes,
                               iou_thrs,
                               phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(
            np.where(gt_has_match)[0],
            gt_boxes[gt_has_match],
            keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate(
                (box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(
                torch.Tensor(gt_box_union[None]),
                torch.Tensor(box_union)).numpy()[0] >= iou_thrs

        else:
            sub_iou = bbox_overlaps(torch.Tensor(gt_box[None, :4]),
                                    torch.Tensor(boxes[:, :4])).numpy()[0]
            obj_iou = bbox_overlaps(torch.Tensor(gt_box[None, 4:]),
                                    torch.Tensor(boxes[:, 4:])).numpy()[0]

            inds = (sub_iou >= iou_thrs) & (obj_iou >= iou_thrs)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def _triplet_panseg(relations,
                    classes,
                    masks,
                    predicate_scores=None,
                    class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        masks (#objs, )
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplet_masks(#rel, 2, , )
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:,
                                                                            2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    masks = np.array(masks)
    triplet_masks = np.stack((masks[sub_id], masks[ob_id]), axis=1)

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id],
            predicate_scores,
            class_scores[ob_id],
        ))

    return triplets, triplet_masks, triplet_scores


def _compute_pred_matches_panseg(gt_triplets,
                                 pred_triplets,
                                 gt_masks,
                                 pred_masks,
                                 iou_thrs,
                                 phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_masks.shape[0])]

    for gt_ind, gt_mask, keep_inds in zip(
            np.where(gt_has_match)[0],
            gt_masks[gt_has_match],
            keeps[gt_has_match],
    ):
        pred_mask = pred_masks[keep_inds]

        sub_gt_mask = gt_mask[0]
        ob_gt_mask = gt_mask[1]
        sub_pred_mask = pred_mask[:, 0]
        ob_pred_mask = pred_mask[:, 1]

        if phrdet:
            # Evaluate where the union mask > 0.5
            inds = []
            gt_mask_union = np.logical_or(sub_gt_mask, ob_gt_mask)
            pred_mask_union = np.logical_or(sub_pred_mask, ob_pred_mask)
            for pred_mask in pred_mask_union:
                iou = mask_iou(gt_mask_union, pred_mask)
                inds.append(iou >= iou_thrs)

        else:
            sub_inds = []
            for pred_mask in sub_pred_mask:
                sub_iou = mask_iou(sub_gt_mask, pred_mask)
                sub_inds.append(sub_iou >= iou_thrs)
            ob_inds = []
            for pred_mask in ob_pred_mask:
                ob_iou = mask_iou(ob_gt_mask, pred_mask)
                ob_inds.append(ob_iou >= iou_thrs)

            inds = np.logical_and(sub_inds, ob_inds)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def mask_iou(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def intersect_2d(x1, x2):
    """Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each
    entry is True if those rows match.

    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError('Input arrays must have same #columns')

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def argsort_desc(scores):
    """Returns the indices that sort scores descending in a smart way.

    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(
        np.unravel_index(np.argsort(-scores.ravel()), scores.shape))



class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""
    def __init__(
        self,
        bboxes=None,  # gt bboxes / OD: det bboxes
        dists=None,  # OD: predicted dists
        labels=None,  # gt labels / OD: det labels
        masks=None,  # gt masks  / OD: predicted masks
        formatted_masks=None,  # OD: Transform the masks for object detection evaluation
        points=None,  # gt points / OD: predicted points
        rels=None,  # gt rel triplets / OD: sampled triplets (training) with target rel labels
        key_rels=None,  # gt key rels
        relmaps=None,  # gt relmaps
        refine_bboxes=None,  # RM: refined object bboxes (score is changed)
        formatted_bboxes=None,  # OD: Transform the refine_bboxes for object detection evaluation
        refine_scores=None,  # RM: refined object scores (before softmax)
        refine_dists=None,  # RM: refined object dists (after softmax)
        refine_labels=None,  # RM: refined object labels
        target_labels=None,  # RM: assigned object labels for training the relation module.
        rel_scores=None,  # RM: predicted relation scores (before softmax)
        rel_dists=None,  # RM: predicted relation prob (after softmax)
        triplet_scores=None,  # RM: predicted triplet scores (the multiplication of sub-obj-rel scores)
        ranking_scores=None,  # RM: predicted ranking scores for rank the triplet
        rel_pair_idxes=None,  # gt rel_pair_idxes / RM: training/testing sampled rel_pair_idxes
        rel_labels=None,  # gt rel_labels / RM: predicted rel labels
        target_rel_labels=None,  # RM: assigned target rel labels
        target_key_rel_labels=None,  # RM: assigned target key rel labels
        saliency_maps=None,  # SAL: predicted or gt saliency map
        attrs=None,  # gt attr
        rel_cap_inputs=None,  # gt relational caption inputs
        rel_cap_targets=None,  # gt relational caption targets
        rel_ipts=None,  # gt relational importance scores
        tgt_rel_cap_inputs=None,  # RM: assigned target relational caption inputs
        tgt_rel_cap_targets=None,  # RM: assigned target relational caption targets
        tgt_rel_ipts=None,  # RM: assigned target relational importance scores
        rel_cap_scores=None,  # RM: predicted relational caption scores
        rel_cap_seqs=None,  # RM: predicted relational seqs
        rel_cap_sents=None,  # RM: predicted relational decoded captions
        rel_ipt_scores=None,  # RM: predicted relational caption ipt scores
        cap_inputs=None,
        cap_targets=None,
        cap_scores=None,
        cap_scores_from_triplet=None,
        alphas=None,
        rel_distribution=None,
        obj_distribution=None,
        word_obj_distribution=None,
        cap_seqs=None,
        cap_sents=None,
        img_shape=None,
        scenes=None,  # gt scene labels
        target_scenes=None,  # target_scene labels
        add_losses=None,  # For Recording the loss except for final object loss and rel loss, e.g.,
        # use in causal head or VCTree, for recording auxiliary loss
        head_spec_losses=None,  # For method-specific loss
        pan_results=None,
    ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all(
            [v is None for k, v in self.__dict__.items() if k != 'self'])

    # HACK: To turn this object into an iterable
    def __len__(self):
        return 1

    # HACK:
    def __getitem__(self, i):
        return self

    # HACK:
    def __iter__(self):
        yield self




def load_results(loadpath):
    with open(loadpath) as infile:
        all_img_dicts = json.load(infile)

    INSTANCE_OFFSET = 1000

    results = []
    for single_result_dict in tqdm(all_img_dicts,
                                   desc='Loading results from json...'):
        pan_seg_filename = single_result_dict['pan_seg_file_name']
        #print(loadpath, 'panseg', pan_seg_filename)
        pan_seg_filename = os.path.join(loadpath, pan_seg_filename)
        pan_seg_img = np.array(Image.open(pan_seg_filename))
        pan_seg_img = pan_seg_img.copy()  # (H, W, 3)
        seg_map = rgb2id(pan_seg_img)

        segments_info = single_result_dict['segments_info']
        num_obj = len(segments_info)

        # get separate masks
        labels = []
        masks = []
        for _, s in enumerate(segments_info):
            label = int(s['category_id'])
            labels.append(label)  # TODO:1-index for gt?
            masks.append(seg_map == s['id'])

        count = dict()
        pan_result = seg_map.copy()
        for _, s in enumerate(segments_info):
            label = int(s['category_id'])
            if label not in count.keys():
                count[label] = 0
            pan_result[seg_map == int(
                s['id']
            )] = label - 1 + count[label] * INSTANCE_OFFSET  # change index?
            count[label] += 1

        rel_array = np.asarray(single_result_dict['relations'])
        if len(rel_array) > 100:
            rel_array = rel_array[:100]
        rel_dists = np.zeros((len(rel_array), 57))
        for idx_rel, rel in enumerate(rel_array):
            rel_dists[idx_rel, rel[2]] += 1  # TODO:1-index for gt?

        result = Result(
            rels=rel_array,
            rel_pair_idxes=rel_array[:, :2],
            masks=masks,
            labels=np.asarray(labels),
            rel_dists=rel_dists,
            refine_bboxes=np.ones((num_obj, 5))
            #pan_results=pan_result,
        )
        results.append(result)

    return results




if __name__=="__main__":

    predicate_cls_list = ['no', 'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of',
                       'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on',
                       'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding',
                       'carrying', 'looking at', 'guiding', 'kissing', 'eating', 'drinking', 'feeding', 'biting',
                       'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching',
                       'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving',
                       'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering',
                       'exiting', 'enclosing', 'leaning on']

    #eval_results = torch.load('psg_eval_results.pytorch', map_location=torch.device("cpu"))
    #predictions = eval_results['predictions']
    #groundtruths = eval_results['groundtruths']

    groundtruths = torch.load('psg_eval_results.pytorch', map_location=torch.device("cpu"))['groundtruths']
    predictions = load_results(os.path.join('.', 'relation.json'))


    sgg_evaluation('sgdet',groundtruths=groundtruths,predictions=predictions,iou_thrs=0.5,logger=None,ind_to_predicates=predicate_cls_list,detection_method = 'pan_seg')
