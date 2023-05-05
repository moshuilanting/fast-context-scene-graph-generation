
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PSG_TEST
@File ：read_dataset.py
@Author ：jintianlei
@Date : 2023/3/6
"""

import json
import numpy as np
import os
from PIL import Image
from ckn.CKN import FC_Net
from vdn.VDN import visual_rel_model
import torchvision.transforms as transforms
import torch
import torchtext
from tqdm import tqdm
import cv2

word2vec_names = ['no','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'hydrant', 'sign',
 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'ball', 'kite', 'bat', 'glove', 'skateboard',
 'surfboard', 'racket', 'bottle', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'plant', 'bed', 'desk', 'toilet',
 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
 'clock', 'vase', 'scissors', 'teddy', 'drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter',
 'curtain', 'door', 'plank', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror', 'net', 'pillow', 'platform',
 'field', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'brick',
 'stone', 'tile', 'wood', 'water', 'blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor',
 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        y2 = mask.shape[0] - 1 if y2 >= mask.shape[0] else y2
        x2 = mask.shape[1] - 1 if x2 >= mask.shape[1] else x2
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.float32)



if __name__=="__main__":

    print('load our model')

    #load vdn
    vdn_model = visual_rel_model('PSG')
    model_dict = vdn_model.state_dict()
    # pretrained_dict = torch.load('runs/0708_EPBS.pt')
    # pretrained_dict = {'image_conv.' + k: v for k, v in pretrained_dict['model'].items() if
    #                    'image_conv.' + k in model_dict}
    # pretrained_dict = torch.load('runs/CKN+EBS.pt', map_location=device)
    pretrained_dict = torch.load('vdn/base_last.pt',map_location='cpu')
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    vdn_model.load_state_dict(model_dict)
    vdn_model = vdn_model.cuda()
    vdn_model.eval()

    total_params = sum(p.numel() for p in vdn_model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in vdn_model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    pretrained_dict = torch.load('last1.pt', map_location='cpu')  # pretrained CKN model
    pretrained_dict = {'image_conv.'+k: v for k, v in pretrained_dict['model'].items() if 'image_conv.'+k in model_dict}
    model_dict.update(pretrained_dict)
    vdn_model.load_state_dict(model_dict)
    vdn_model = vdn_model.cuda()
    vdn_model.eval()


    ckn_model = FC_Net('PSG')
    model_dict = ckn_model.state_dict()
    pretrained_dict = torch.load('last1.pt',map_location='cpu') # pretrained CKN model
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    ckn_model.load_state_dict(model_dict)
    ckn_model = ckn_model.cuda()
    ckn_model.eval()

    # load word vector
    word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
    word_vec_list = []
    for name_index in range(len(word2vec_names)):
        word_vec = word2vec.get_vecs_by_tokens(word2vec_names[name_index], lower_case_backup=True)
        word_vec_list.append(word_vec)
    word_vec_list = torch.stack(word_vec_list)

    loadpath = 'SegFromer_PVTV'


    with open('dataset/psg_val_test.json','r') as infile:
        val_img_dicts = json.load(infile)['data']


    with open(os.path.join('SegFromer_PVTV','trans_seg_annotations.json'),'r') as f:
        all_img_dicts = json.load(f)

    image_path = '/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/OpenPSG/data'

    find_img_dicts = []
    for i in range(len(val_img_dicts)):
        if len(val_img_dicts[i]['relations'])==0:
            continue
        file_name = val_img_dicts[i]['pan_seg_file_name'].split('/')[1]
        for j in range(len(all_img_dicts)):
            ergodic_file_name = all_img_dicts[j]['file_name']
            if file_name==ergodic_file_name:
                #print(file_name)
                find_img_dicts.append(all_img_dicts[j])


    new_all_img_dicts = []
    count = 0
    with torch.no_grad():
        for single_result_dict in tqdm(find_img_dicts,
                                       desc='Loading results from json...'):
            pan_seg_filename = single_result_dict['file_name']
            image_filename = os.path.join(image_path, 'coco', 'val2017', pan_seg_filename.split('.')[0] + '.jpg')
            image = cv2.imread(image_filename)

            pan_seg_filename = os.path.join(loadpath, 'seg', pan_seg_filename)
            pan_seg_img = Image.open(pan_seg_filename)

            single_result_dict['pan_seg_file_name'] = pan_seg_filename
            count += 1
            pan_seg_img = np.array(pan_seg_img)
            pan_seg_img = pan_seg_img.copy()  # (H, W, 3)


            seg_map = rgb2id(pan_seg_img)
            height, width, channel = pan_seg_img.shape

            segments_info = single_result_dict['segments_info']
            # print(single_result_dict['pan_seg_file_name'],len(single_result_dict['segments_info']))
            num_obj = len(segments_info)

            # get separate masks
            labels = []
            masks = []
            for _, s in enumerate(segments_info):
                label = int(s['category_id'])
                labels.append(label)  # TODO:1-index for gt?
                masks.append(seg_map == s['id'])

            current_categroy = np.array(labels)
            current_mask = np.stack(masks, axis=2)
            current_bbox_hw = extract_bboxes(current_mask)

            # realtion detection
            current_bbox_hw = torch.from_numpy(current_bbox_hw)
            current_bbox = current_bbox_hw / (torch.tensor([width, height, width, height]))  # normalization

            current_categroy = torch.from_numpy(current_categroy)
            obj_location_feature = torch.cat((((current_bbox[:, 2] - current_bbox[:, 0]) / 2).unsqueeze(1),
                                              ((current_bbox[:, 3] - current_bbox[:, 1]) / 2).unsqueeze(1),
                                              current_bbox), 1)
            obj_word_feature = word_vec_list[current_categroy.long()]
            # print(obj_word_feature.size(),obj_location_feature.size())            #obj = torch.cat((obj_word_feature, obj_location_feature), 1)
            obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)
            obj_id = torch.arange((len(obj_location_feature)))
            a_id = obj_id.repeat(len(obj_id), 1).view(-1, )
            b_id = torch.repeat_interleave(obj_id, len(obj_id), dim=0)
            obj_pair = torch.cat((a_id.unsqueeze(1), b_id.unsqueeze(1)), dim=1)
            obj_pair = obj_pair[obj_pair[:, 0] != obj_pair[:, 1]]

            # [word_vector,c_x,c_y,x1,y1,x2,y2 ,word_vector,c_x,c_y,x1,y1,x2,y2, dc_x, dc_y, dx1,dx2,dy2,dy2]
            rel_feature = torch.cat((obj[obj_pair[:, 0].long()], obj[obj_pair[:, 1].long()],
                                     obj[obj_pair[:, 1].long()][:, 50:] - obj[obj_pair[:, 0].long()][:, 50:]), 1)
            rel_feature = rel_feature.float().cuda()

            y = ckn_model(rel_feature)
            pred_relations = torch.sigmoid(y)

            pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]
            a, indices = torch.sort(pred_relations_conf.view(-1), descending=True)
            # a = torch.sort(pred_relations_conf, dim=1)

            select_obj_pair = obj_pair[indices[:100]]
            select_rel_feature = rel_feature[indices[:100]]
            pred_relations_conf = pred_relations_conf[indices[:100]]
            pred_relations = pred_relations[indices[:100]]
            pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations],
                                       dim=1)
            # print(len(select_obj_pair))

            if len(select_obj_pair) > 1:
                visual_bbox_pair = torch.cat((current_bbox_hw[select_obj_pair[:, 0].long()][:, :4],
                                              current_bbox_hw[select_obj_pair[:, 1].long()][:, :4]), 1)
                cut_x1, _ = torch.min(visual_bbox_pair[:, [0, 4]], dim=1)
                cut_y1, _ = torch.min(visual_bbox_pair[:, [1, 5]], dim=1)
                cut_x2, _ = torch.max(visual_bbox_pair[:, [2, 6]], dim=1)
                cut_y2, _ = torch.max(visual_bbox_pair[:, [3, 7]], dim=1)

                batch_visual_pred = []
                batch_semantic_pred = []
                visual_bbox_pair = torch.split(visual_bbox_pair, 50, dim=0)
                _rel_feature = torch.split(select_rel_feature, 50, dim=0)

                for j in range(len(visual_bbox_pair)):
                    batch_img = []
                    current_batch_visual_bbox_pair = visual_bbox_pair[j]
                    batch_word_feature = _rel_feature[j]
                    for i in range(len(current_batch_visual_bbox_pair)):
                        x1, y1, x2, y2 = int(cut_x1[j * 50 + i]), int(cut_y1[j * 50 + i]), int(cut_x2[j * 50 + i]), int(
                            cut_y2[j * 50 + i])
                        img = image[y1:y2, x1:x2]
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                        source_mask = masks[obj_pair[j * 50 + i][0]].astype(np.float32)
                        target_mask = masks[obj_pair[j * 50 + i][1]].astype(np.float32)

                        source_mask = source_mask[y1:y2, x1:x2]
                        target_mask = target_mask[y1:y2, x1:x2]

                        source_mask = cv2.resize(source_mask, (224, 224), interpolation=cv2.INTER_AREA)
                        target_mask = cv2.resize(target_mask, (224, 224), interpolation=cv2.INTER_AREA)

                        img = transforms.ToTensor()(img)
                        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                        source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                        target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                        batch_img.append(torch.cat((source_mask, img, target_mask), 0))

                    batch_img = torch.stack(batch_img).cuda()
                    visual_feature, semantic_pred, fusion_relation_pred = vdn_model(batch_img, batch_word_feature)
                    batch_visual_pred.append(fusion_relation_pred)
                    batch_semantic_pred.append(semantic_pred)

                pred_fusion_relations = torch.cat(batch_visual_pred, 0)
                pred_semantic_relations = torch.cat(batch_semantic_pred, 0)

                _pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]
                pred_relations = torch.cat(
                    [torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations], dim=1)
                pred_relations_values, pred_relation_indices = torch.sort(pred_relations, 1, descending=True)
                indices = torch.arange(0, len(pred_relations_values)).cuda()
                _indices_neg = torch.repeat_interleave(indices, len(pred_relation_indices[:, 10:][0]), dim=0)
                _pred_relation_indices_neg = torch.cat(
                    (_indices_neg.reshape(-1, 1), pred_relation_indices[:, 10:].reshape(-1, 1)), 1)
                _indices_pos = torch.repeat_interleave(indices, len(pred_relation_indices[:, :10][0]), dim=0)
                _pred_relation_indices_pos = torch.cat(
                    (_indices_pos.reshape(-1, 1), pred_relation_indices[:, :10].reshape(-1, 1)), 1)
                pred_relations[_pred_relation_indices_neg[:, 0], _pred_relation_indices_neg[:, 1]] = 0.0

                _pred_relations_conf, pred_relations = pred_semantic_relations[:, 0:1], pred_fusion_relations[:, 1:]

                pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations],dim=1)
                pred_relations[_pred_relation_indices_neg[:, 0], _pred_relation_indices_neg[:, 1]] = 0.0


            relations = [[0, 0, 0]]
            if len(pred_relations) > 0:

                pred_relations = pred_relations + torch.from_numpy(relation_class_mask * 0.3).cuda()
      
                relation_max, relation_argmax = torch.max(pred_relations, dim=1)
                # confidence sort
                # head_semantic = torch.cat([obj_pair[:, :2].cuda(), relation_argmax.view(-1, 1), relation_max.view(-1, 1), pred_relations],dim=1)
                head_semantic = torch.cat(
                    [select_obj_pair[:, :2].cuda(), relation_argmax.view(-1, 1), pred_relations_conf.view(-1, 1),
                     pred_relations], dim=1)

                head_semantic = head_semantic[torch.argsort(head_semantic[:, 3], descending=True)]

                # condition 1. >0.1 and not two stuffs; 2.
                raw_rels = head_semantic[:, :4].cpu().detach().numpy()
                relations = raw_rels[:, :3].astype(np.int32).tolist()

            single_result_dict = dict(
                relations=relations,
                segments_info=single_result_dict['segments_info'],
                # pan_seg_file_name=raw_data['data'][i]["file_name"].split('/')[1].split('.')[0] +'.png',
                pan_seg_file_name=single_result_dict['pan_seg_file_name']
            )

            new_all_img_dicts.append(single_result_dict)

    with open(os.path.join('relation.json'), 'w') as outfile:
        json.dump(new_all_img_dicts, outfile, default=str)
