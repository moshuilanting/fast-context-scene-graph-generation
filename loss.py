#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：623
@File ：loss.py
@Author ：jintianlei
@Date : 2022/6/24
"""
import torch
import torch.nn as nn
import numpy as np

class SGG_ComputeLoss:
    # Compute losses
    def __init__(self,device,dataset_name='VG'):
        self.device = device
        self.dataset_name = dataset_name
        self.BCErel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))

    def __call__(self, p, rel_targets):  # predictions, targets
        if self.dataset_name == 'VG':
            target_vector = torch.eye(51)[rel_targets[:, 1].long()].to(self.device)
        elif self.dataset_name == 'PSG':
            target_vector = torch.eye(57)[rel_targets[:, 1].long()].to(self.device)
        elif self.dataset_name == 'OID':
            target_vector = torch.eye(31)[rel_targets[:, 1].long()].to(self.device)

        conf_target = (1-target_vector[:,0])

        conf_loss = self.BCEobj(p[:, 0], conf_target)

        conf_target = conf_target.bool()
        relation_target = target_vector[conf_target][:,1:]

        relation_loss = self.BCErel(p[conf_target][:,1:], relation_target)

        return relation_loss,conf_loss


class Visual_Contrast_ComputeLoss:
    # Compute losses
    def __init__(self, device, sample_number, dataset_name= 'OID'):
        self.device = device
        self.sample_num = sample_number
        self.dataset_name = dataset_name
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        self.fusion_BCErel = nn.BCELoss()
        self.fusion_BCEobj = nn.BCELoss()

        self.semantic_BCErel = nn.BCELoss()
        self.semantic_BCEobj = nn.BCELoss()



    def __call__(self, visual_feature, semantic_pred, fusion_relation_pred, batch_label):  # predictions, targets
        pos_batch_label = batch_label[batch_label[:, 0] == 1]

        if self.dataset_name == 'VG':
            target_vector = torch.eye(51)[batch_label[:, 1].long()].to(self.device)
        elif self.dataset_name == 'PSG':
            target_vector = torch.eye(57)[batch_label[:, 1].long()].to(self.device)
        elif self.dataset_name == 'OID':
            target_vector = torch.eye(31)[batch_label[:, 1].long()].to(self.device)

        conf_target = (1 - target_vector[:, 0])


        fusion_conf_loss = self.fusion_BCEobj(fusion_relation_pred[:, 0], conf_target)

        conf_target = conf_target.bool()


        pos_fusion_relation_pred = fusion_relation_pred[batch_label[:, 0] == 1]
        pos_semantic_pred = semantic_pred[batch_label[:, 0] == 1]


        pos_semantic_pred_relations_values, pos_semantic_pred_indices = torch.sort(pos_semantic_pred[:, 1:], 1,descending=True)

        current_top_k_mask = torch.zeros((len(pos_semantic_pred_relations_values), 31), device=self.device)

        indices = torch.arange(0, len(pos_semantic_pred_indices), device=self.device)
        current_top_k_mask[:, 1:][indices.unsqueeze(1), pos_semantic_pred_indices[:, :10]] = 1  # top 10

        epsilon = 0.1

        pos_boost_loss = torch.relu(((pos_semantic_pred-pos_fusion_relation_pred) + epsilon)[indices,pos_batch_label[:,1].long()]).mean()
        neg_suppress_loss = torch.relu(((pos_fusion_relation_pred-pos_semantic_pred) + epsilon) *current_top_k_mask).sum() / current_top_k_mask.sum()

        return fusion_conf_loss, pos_boost_loss, neg_suppress_loss
