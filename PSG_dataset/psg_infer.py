
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
import torch
import torchtext
from tqdm import tqdm

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
    rel_model = FC_Net('PSG')
    model_dict = rel_model.state_dict()
    # pretrained_dict = torch.load('runs/0708_EPBS.pt')
    # pretrained_dict = {'image_conv.' + k: v for k, v in pretrained_dict['model'].items() if
    #                    'image_conv.' + k in model_dict}
    # pretrained_dict = torch.load('runs/CKN+EBS.pt', map_location=device)
    pretrained_dict = torch.load('ckn/EPBS_PSG_last.pt', map_location='cuda:0')
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    rel_model.load_state_dict(model_dict)
    rel_model = rel_model.cuda()
    rel_model.eval()

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
    for single_result_dict in tqdm(find_img_dicts,desc='Start reasoning and save the results to submission_seg...'):
        pan_seg_filename = single_result_dict['file_name']
        pan_seg_filename = os.path.join(loadpath, 'seg', pan_seg_filename)
        pan_seg_img = Image.open(pan_seg_filename)
        #pan_seg_img.save(os.path.join(savepath, 'panseg', '{:d}.png'.format(count)))
        single_result_dict['pan_seg_file_name'] = pan_seg_filename #'{:d}.png'.format(count)
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
        current_bbox = extract_bboxes(current_mask)

        # realtion detection
        current_bbox = torch.from_numpy(current_bbox)
        current_bbox /= torch.tensor([width, height, width, height])  # normalization

        current_categroy = torch.from_numpy(current_categroy)
        obj_location_feature = torch.cat((((current_bbox[:, 2] - current_bbox[:, 0]) / 2).unsqueeze(1),
                                          ((current_bbox[:, 3] - current_bbox[:, 1]) / 2).unsqueeze(1),
                                          current_bbox), 1)
        obj_word_feature = word_vec_list[current_categroy.long()]

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

        y = rel_model(rel_feature)
        pred_relations = torch.sigmoid(y)

        pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]

        pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations],
                                   dim=1)


        relations = [[0, 0, 0]]
        if len(pred_relations) > 0:
            # print(pred_relations[1])
            pred_relations = pred_relations + torch.from_numpy(relation_class_mask).cuda()

            relation_max, relation_argmax = torch.max(pred_relations, dim=1)

            pred_relations_conf = pred_relations_conf.view(-1) * relation_max

            # confidence sort
            # head_semantic = torch.cat([obj_pair[:, :2].cuda(), relation_argmax.view(-1, 1), relation_max.view(-1, 1), pred_relations],dim=1)
            head_semantic = torch.cat(
                [obj_pair[:, :2].cuda(), relation_argmax.view(-1, 1), pred_relations_conf.view(-1, 1), pred_relations],
                dim=1)

            head_semantic = head_semantic[torch.argsort(head_semantic[:, 3], descending=True)]

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
