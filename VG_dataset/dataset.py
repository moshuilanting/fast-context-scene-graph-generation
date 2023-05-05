#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：623
@File ：dataset.py
@Author ：jintianlei
@Date : 2022/6/23
"""
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import cv2
import torch
import numpy as np
import base64
import pickle
import random
import torchtext
import torchvision.transforms as transforms








names = ['no','airplane','animal','arm','bag','banana','basket','beach','bear','bed','bench','bike','bird','board','boat','book','boot','bottle','bowl','box','boy','branch','building','bus','cabinet','cap','car','cat','chair','child','clock','coat','counter','cow','cup','curtain','desk','dog','door','drawer','ear','elephant','engine','eye','face','fence','finger','flag','flower','food','fork','fruit','giraffe','girl','glass','glove','guy','hair','hand','handle','hat','head','helmet','hill','horse','house','jacket','jean','kid','kite','lady','lamp','laptop','leaf','leg','letter','light','logo','man','men','motorcycle','mountain','mouth','neck','nose','number','orange','pant','paper','paw','people','person','phone','pillow','pizza','plane','plant','plate','player','pole','post','pot','racket','railing','rock','roof','room','screen','seat','sheep','shelf','shirt','shoe','short','sidewalk','sign','sink','skateboard','ski','skier','sneaker','snow','sock','stand','street','surfboard','table','tail','tie','tile','tire','toilet','towel','tower','track','train','tree','truck','trunk','umbrella','vase','vegetable','vehicle','wave','wheel','window','windshield','wing','wire','woman','zebra']  # class names
relations_names = ['no','above','across','against','along','and','at','attached to','behind','belonging to','between','carrying','covered in','covering','eating','flying in','for','from','growing on','hanging from','has','holding','in','in front of','laying on','looking at','lying on','made of','mounted on','near','of','on','on back of','over','painted on','parked on','part of','playing','riding','says','sitting on','standing on','to','under','using','walking in','walking on','watching','wearing','wears','with']

def get_mask_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution,resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    #head_channel = torch.from_numpy(head_channel)
    return head_channel


class Scene_Graph(Dataset):
    def __init__(self, object_label_dir,relation_label_dir, group=20, sample_num=10, sampling='EBS',training = True):
        self.training = training
        self.sampling = sampling
        self.group = group # each batch 20 self.group from 50 relations
        self.sample_num = sample_num # each class have self.sample_num pos samples and self.sample_num neg samples

        word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
        self.word_vec_list = []
        for name_index in range(len(names)):
            word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
            self.word_vec_list.append(word_vec)
        self.word_vec_list = torch.stack(self.word_vec_list)
        #print(self.word_vec_list)

        rootdir = os.path.join(object_label_dir)
        self.object_label_files = []
        self.relation_label_files = []

        for (dirpath, dirnames, filenames) in os.walk(rootdir):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.txt':
                    self.object_label_files.append(os.path.join(object_label_dir, filename))

                    relation_file = os.path.join(relation_label_dir, filename)
                    self.relation_label_files.append(relation_file)

        self.object_label_files = sorted(self.object_label_files)
        self.relation_label_files = sorted(self.relation_label_files)

        if os.path.isfile('relation.npy') and os.path.isfile('object.npy'):
            self.object_labels = np.load('object.npy',allow_pickle=True)
            self.relation_labels = np.load('relation.npy',allow_pickle=True)

        else:
            self.object_labels = []
            self.relation_labels = []
            for i in range(len(self.object_label_files)):
                with open(self.object_label_files[i], 'r') as f1:
                    l1 = [x.split() for x in f1.read().strip().splitlines() if len(x)]
                    l1 = np.array(l1,dtype=np.float32)
                    self.object_labels.append(l1)

                with open(self.relation_label_files[i], 'r') as f2:
                    l2 = [x.split() for x in f2.read().strip().splitlines() if len(x)]
                    l2 = np.array(l2, dtype=np.float32)
                    self.relation_labels.append(l2)
                if i %1000==0:
                    print('{}/{}'.format(i,len(self.object_label_files)))

            self.object_labels = np.array(self.object_labels)
            self.relation_labels = np.array(self.relation_labels)

            np.save('object.npy',self.object_labels)
            np.save('relation.npy', self.relation_labels)

        print('finish read trainning data: ', len(self.object_label_files))

        self.object_pair_labels = []
        self.object_pair_labels_pool = []
        if self.training == True:
            for i in range(len(self.relation_labels)):
                l1 = torch.from_numpy(self.object_labels[i])
                l2 = torch.from_numpy(self.relation_labels[i])

                #img_id, source_id, target_id, source_cls,source_bbox,target_cls,target_bbox,predicate
                img_id = torch.tensor([i for j in range(len(l2))]).unsqueeze(1)
                source_id = l2[:, 0].long()
                target_id = l2[:, 1].long()

                source_cls = l1[source_id][:,0:1]
                target_cls = l1[target_id][:,0:1]
                source_bbox = l1[source_id][:, 1:]
                target_bbox = l1[target_id][:, 1:]
                source_id = l2[:, 0:1].long()
                target_id = l2[:, 1:2].long()
                predicate = l2[:, 2:3]

                object_pair = torch.cat((img_id, source_id, target_id, source_cls,source_bbox,target_cls,target_bbox,predicate), 1)
                self.object_pair_labels.append(object_pair)

            self.object_pair_labels = torch.cat(self.object_pair_labels,0)
            print(self.sampling + 'trainning samples: ', len(self.object_pair_labels), )

            for i in range(1,51):
                self.object_pair_labels_pool.append(self.object_pair_labels[self.object_pair_labels[:,-1]==i])
            #
            for i in range(len(self.object_pair_labels_pool)):
                print(i+1,relations_names[i+1], len(self.object_pair_labels_pool[i]))


    def __getitem__(self, index):

        if self.training == True:
            if self.sampling == 'PBS':
                current_batch = []
                current_group = torch.from_numpy(np.random.randint(1, len(relations_names) ,size = self.group))
                for i in range(self.group):
                    current_predcls = self.object_pair_labels[self.object_pair_labels[:, -1] == current_group[i]]
                    current_samples = torch.from_numpy(np.random.randint(0, len(current_predcls) ,size = self.sample_num))
                    current_predcls = current_predcls[current_samples]
                    current_batch.append(current_predcls)
                current_batch = torch.cat(current_batch,0)

            elif self.sampling == 'EBS':
                current_batch = []
                current_group = torch.from_numpy(np.random.randint(1, len(names), size=self.group))
                for i in range(self.group):
                    current_entity = self.object_pair_labels[(
                                (self.object_pair_labels[:, 3] == current_group[i]).int() | (
                                    self.object_pair_labels[:, 8] == current_group[i]).int()).bool()]
                    current_samples = torch.from_numpy(np.random.randint(0, len(current_entity), size=self.sample_num))
                    current_entity = current_entity[current_samples]
                    current_batch.append(current_entity)
                current_batch = torch.cat(current_batch, 0)

            else:
                if index % 2 == 0:
                    #entity balanced sampling
                    current_batch = []
                    current_group = torch.from_numpy(np.random.randint(1, len(names), size=self.group))
                    for i in range(self.group):
                        current_entity = self.object_pair_labels[(
                                    (self.object_pair_labels[:, 3] == current_group[i]).int() | (
                                        self.object_pair_labels[:, 8] == current_group[i]).int()).bool()]
                        current_samples = torch.from_numpy(
                            np.random.randint(0, len(current_entity), size=self.sample_num))
                        current_entity = current_entity[current_samples]
                        current_batch.append(current_entity)
                    current_batch = torch.cat(current_batch, 0)
                else:
                    # predcl balanced sampling
                    current_batch = []
                    current_group = torch.from_numpy(np.random.randint(1, len(relations_names), size=self.group))
                    for i in range(self.group):
                        current_predcls = self.object_pair_labels[self.object_pair_labels[:, -1] == current_group[i]]
                        current_samples = torch.from_numpy(np.random.randint(0, len(current_predcls), size=self.sample_num))
                        current_predcls = current_predcls[current_samples]
                        current_batch.append(current_predcls)
                    current_batch = torch.cat(current_batch, 0)


            source_word_feature = self.word_vec_list[current_batch[:,3].long()]
            source_bbox = current_batch[:,4:8]
            nosiy_location = torch.randn((len(source_bbox),4), out=None)*0.1

            source_bbox = source_bbox+nosiy_location
            source_bbox = torch.cat((((source_bbox[:, 2] - source_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((source_bbox[:, 3] - source_bbox[:, 1]) / 2).unsqueeze(1), source_bbox), 1)

            traget_word_feature = self.word_vec_list[current_batch[:,8].long()]
            traget_bbox = current_batch[:,9:13]
            nosiy_location = torch.randn((len(traget_bbox),4), out=None)*0.1

            traget_bbox = traget_bbox+nosiy_location
            traget_bbox = torch.cat((((traget_bbox[:, 2] - traget_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((traget_bbox[:, 3] - traget_bbox[:, 1]) / 2).unsqueeze(1), traget_bbox), 1)
            rel_label = torch.cat((torch.tensor([0 for i in range(len(current_batch))]).unsqueeze(1), current_batch[:, 13].unsqueeze(1)), 1)


            source_feature =  torch.cat((source_word_feature, source_bbox.repeat((1, 5))), 1)
            target_feature = torch.cat((traget_word_feature, traget_bbox.repeat((1, 5))), 1)

            rel_feature = torch.cat((source_feature,target_feature,target_feature[:,50:]-source_feature[:,50:]),1)

            # 生成无关系的样本
            img_id = np.random.randint(0, len(self.relation_labels), size=self.group)
            for i in range(len(img_id)):
                l1 = torch.from_numpy(self.object_labels[img_id[i]])
                l2 = torch.from_numpy(self.relation_labels[img_id[i]])
                obj_num = len(l1)
                source_id = np.random.randint(0, obj_num, size=self.sample_num)
                target_id = np.random.randint(0, obj_num, size=self.sample_num)
                s_t = np.concatenate((source_id.reshape(-1, 1), target_id.reshape(-1, 1)), 1)
                no_relation_index = []
                for i in range(len(s_t)):
                    if s_t[i].tolist() not in l2[:, :2].tolist():
                        no_relation_index.append(s_t[i])
                no_relation_index = torch.from_numpy(np.array(no_relation_index))

                # [word_vector,c_x,c_y,x1,y1,x2,y2]
                obj_location_feature = torch.cat(
                    (((l1[:, 3] - l1[:, 1]) / 2).unsqueeze(1), ((l1[:, 4] - l1[:, 2]) / 2).unsqueeze(1), l1[:, 1:]), 1)
                obj_word_feature = self.word_vec_list[l1[:, 0].long()]

                obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)

                if len(no_relation_index) > 1:
                    no_rel_feature = torch.cat((obj[no_relation_index[:, 0].long()], obj[no_relation_index[:, 1].long()],
                                                obj[no_relation_index[:, 1].long()][:, 50:] - obj[no_relation_index[:,
                                                                                                  0].long()][:, 50:]), 1)
                    no_rel_label = torch.cat((torch.tensor([0 for i in range(len(no_relation_index))]).unsqueeze(1),
                                              torch.tensor([0 for i in range(len(no_relation_index))]).unsqueeze(1)), 1)

                    rel_feature = torch.cat((rel_feature, no_rel_feature), 0)
                    rel_label = torch.cat((rel_label, no_rel_label), 0)

            return rel_feature,rel_label

        else:
            l1 = torch.from_numpy(self.object_labels[index])
            l2 = torch.from_numpy(self.relation_labels[index])

            obj_location_feature =  torch.cat(( ((l1[:,3]-l1[:,1])/2).unsqueeze(1),((l1[:,4]-l1[:,2])/2).unsqueeze(1),l1[:,1:]),1)
            obj_word_feature = self.word_vec_list[l1[:,0].long()]
            obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)

            rel_feature = torch.cat((obj[l2[:, 0].long()], obj[l2[:, 1].long()],obj[l2[:, 1].long()][:, 50:] - obj[l2[:, 0].long()][:, 50:]), 1)
            rel_label = torch.cat((torch.tensor([0 for i in range(len(rel_feature))]).unsqueeze(1), l2[:, 2].unsqueeze(1)), 1)
            return rel_feature, rel_label, l1, l2

    def __len__(self):
        if self.training==True:
            return len(self.object_pair_labels)//(self.group*self.sample_num)
        else:
            return len(self.relation_label_files)


    # def collate_fn(self,batch):
    #     if self.training == True:
    #         rel_feature,rel_label = zip(*batch)  # transposed
    #
    #         # for i, l in enumerate(rel_feature):
    #         #     l[:, 0] = i  # add target image index for build_targets()
    #
    #         for i, l in enumerate(rel_label):
    #             l[:, 0] = i  # add target image index for build_targets()
    #
    #         return torch.cat(rel_feature, 0), torch.cat(rel_label, 0)
    #     else:
    #         rel_feature, rel_label, object_targets, relation_target = zip(*batch)  # transposed
    #
    #         # for i, l in enumerate(rel_feature):
    #         #     l[:, 0] = i  # add target image index for build_targets()
    #
    #         for i, l in enumerate(rel_label):
    #             l[:, 0] = i  # add target image index for build_targets()
    #
    #         return torch.cat(rel_feature, 0), torch.cat(rel_label, 0),torch.cat(object_targets, 0),torch.cat(relation_target, 0)


class Scene_Graph_Image(Dataset):
    def __init__(self, image_dir, object_label_dir, relation_label_dir, group=20, sample_num=10, sampling = 'EPBS' , training=True):
        self.training = training
        self.group = group
        self.sample_num = sample_num
        self.sampling = sampling
        rootdir = os.path.join(object_label_dir)
        self.object_label_files = []
        self.relation_label_files = []
        self.image_files = []

        word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
        self.word_vec_list = []
        for name_index in range(len(names)):
            word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
            self.word_vec_list.append(word_vec)
        self.word_vec_list = torch.stack(self.word_vec_list)

        for (dirpath, dirnames, filenames) in os.walk(rootdir):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.txt':
                    self.object_label_files.append(os.path.join(object_label_dir, filename))

                    relation_file = os.path.join(relation_label_dir, filename)
                    self.relation_label_files.append(relation_file)

                    self.image_files.append(filename.split('.')[0] + '.jpg')

        self.object_label_files = sorted(self.object_label_files)
        self.relation_label_files = sorted(self.relation_label_files)
        self.image_files = sorted(self.image_files)

        if os.path.isfile('relation.npy') and os.path.isfile('object.npy') and os.path.isfile('image.pkl'):
            self.object_labels = np.load('object.npy', allow_pickle=True)
            self.relation_labels = np.load('relation.npy', allow_pickle=True)
            # self.image_data = np.load('image.npy', allow_pickle=True)
            self.image_data = pickle.load(open('image.pkl', 'rb'))

        else:
            self.object_labels = []
            self.relation_labels = []
            self.image_data = []
            for i in range(len(self.object_label_files)):
                with open(self.object_label_files[i], 'r') as f1:
                    l1 = [x.split() for x in f1.read().strip().splitlines() if len(x)]
                    l1 = np.array(l1, dtype=np.float32)
                    self.object_labels.append(l1)

                with open(self.relation_label_files[i], 'r') as f2:
                    l2 = [x.split() for x in f2.read().strip().splitlines() if len(x)]
                    l2 = np.array(l2, dtype=np.float32)
                    self.relation_labels.append(l2)
                if i % 1000 == 0:
                    print('{}/{}'.format(i, len(self.object_label_files)))

                assert os.path.exists(os.path.join(image_dir, self.image_files[i])), 'no such image {:s}'.format(
                    os.path.join(image_dir, self.image_files[i]))
                # img = cv2.imread(os.path.join(image_dir, self.image_files[i]))
                self.image_data.append(os.path.join(image_dir, self.image_files[i]))

            self.object_labels = np.array(self.object_labels)
            self.relation_labels = np.array(self.relation_labels)
            # self.image_data = np.array(self.image_data)

            np.save('object.npy', self.object_labels)
            np.save('relation.npy', self.relation_labels)
            # np.save('image.npy', self.image_data)
            pickle.dump(self.image_data, open('image.pkl', 'wb'), protocol=4)

            print('finish read trainning data: ', len(self.object_label_files))

        self.object_pair_labels = []

        for i in range(len(self.relation_labels)):
            l1 = torch.from_numpy(self.object_labels[i])
            l2 = torch.from_numpy(self.relation_labels[i])

            # img_id, source_id, target_id, source_cls,source_bbox,target_cls,target_bbox,predicate
            img_id = torch.tensor([i for j in range(len(l2))]).unsqueeze(1)
            source_id = l2[:, 0].long()
            target_id = l2[:, 1].long()

            source_cls = l1[source_id][:, 0:1].int()
            target_cls = l1[target_id][:, 0:1].int()
            source_bbox = l1[source_id][:, 1:]
            target_bbox = l1[target_id][:, 1:]
            source_id = l2[:, 0:1].int()
            target_id = l2[:, 1:2].int()
            predicate = l2[:, 2:3].int()

            object_pair = torch.cat(
                (img_id, source_id, target_id, source_cls, source_bbox, target_cls, target_bbox, predicate), 1)
            self.object_pair_labels.append(object_pair)

        self.object_pair_labels = torch.cat(self.object_pair_labels, 0)
        print('finish read samples data: ', len(self.object_pair_labels), self.sampling)

        print(self.object_pair_labels.size())

    def __getitem__(self, index):
        if self.training == True:
            if self.sampling == 'EBS':
                # entity balanced sampling
                current_batch = []
                current_group = torch.from_numpy(np.random.randint(1, len(names), size=self.group))
                for i in range(self.group):
                    current_entity = self.object_pair_labels[(
                            (self.object_pair_labels[:, 3] == current_group[i]).int() | (
                            self.object_pair_labels[:, 8] == current_group[i]).int()).bool()]
                    current_samples = torch.from_numpy(np.random.randint(0, len(current_entity), size=self.sample_num))
                    current_entity = current_entity[current_samples]
                    current_batch.append(current_entity)
                current_batch = torch.cat(current_batch, 0)

            else:
                if index % 2 == 0:
                    # predcl balanced sampling
                    current_batch = []
                    # current_group = torch.from_numpy(np.random.randint(1, len(relations_names), size=self.group))
                    current_group = torch.from_numpy(np.array(random.sample(range(1, len(relations_names)), self.group)))
                    for i in range(self.group):
                        current_predcls = self.object_pair_labels[self.object_pair_labels[:, 13] == current_group[i]]
                        current_samples = torch.from_numpy(np.random.randint(0, len(current_predcls), size=self.sample_num))
                        current_predcls = current_predcls[current_samples]
                        current_batch.append(current_predcls)
                    current_batch = torch.cat(current_batch, 0)

                else:
                    # entity balanced sampling
                    current_batch = []
                    current_group = torch.from_numpy(np.random.randint(1, len(names), size=self.group))
                    for i in range(self.group):
                        current_entity = self.object_pair_labels[(
                                (self.object_pair_labels[:, 3] == current_group[i]).int() | (
                                self.object_pair_labels[:, 8] == current_group[i]).int()).bool()]
                        current_samples = torch.from_numpy(np.random.randint(0, len(current_entity), size=self.sample_num))
                        current_entity = current_entity[current_samples]
                        current_batch.append(current_entity)
                    current_batch = torch.cat(current_batch, 0)



            #  #random sampling
            #  current_index = torch.from_numpy(np.random.randint(0, len(self.object_pair_labels), size=self.group*self.sample_num))
            #  current_batch = self.object_pair_labels[current_index]

            # raw image data generate
            cut_x1, _ = torch.min(current_batch[:, [4, 9]], dim=1)
            cut_y1, _ = torch.min(current_batch[:, [5, 10]], dim=1)
            cut_x2, _ = torch.max(current_batch[:, [6, 11]], dim=1)
            cut_y2, _ = torch.max(current_batch[:, [7, 12]], dim=1)

            cut_x1 = torch.clip(cut_x1 - torch.rand(len(cut_x1)) * 0.1, 0, 1)
            cut_y1 = torch.clip(cut_y1 - torch.rand(len(cut_x1)) * 0.1, 0, 1)
            cut_x2 = torch.clip(cut_x2 + torch.rand(len(cut_x1)) * 0.1, 0, 1)
            cut_y2 = torch.clip(cut_y2 + torch.rand(len(cut_x1)) * 0.1, 0, 1)

            source_word_feature = self.word_vec_list[current_batch[:, 3].long()]
            source_bbox = current_batch[:, 4:8]
            nosiy_location = torch.randn((len(source_bbox), 4), out=None) * 0.1

            source_bbox = source_bbox + nosiy_location
            source_bbox = torch.cat((((source_bbox[:, 2] - source_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((source_bbox[:, 3] - source_bbox[:, 1]) / 2).unsqueeze(1), source_bbox), 1)
            traget_word_feature = self.word_vec_list[current_batch[:, 8].long()]
            traget_bbox = current_batch[:, 9:13]
            nosiy_location = torch.randn((len(traget_bbox), 4), out=None) * 0.1

            traget_bbox = traget_bbox + nosiy_location
            traget_bbox = torch.cat((((traget_bbox[:, 2] - traget_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((traget_bbox[:, 3] - traget_bbox[:, 1]) / 2).unsqueeze(1), traget_bbox), 1)
            batch_label = torch.cat(
                (torch.tensor([1 for i in range(len(current_batch))]).unsqueeze(1), current_batch[:, 13].unsqueeze(1)),
                1)

            source_feature = torch.cat((source_word_feature, source_bbox.repeat((1, 5))), 1)
            target_feature = torch.cat((traget_word_feature, traget_bbox.repeat((1, 5))), 1)

            batch_word_feature = torch.cat(
                (source_feature, target_feature, target_feature[:, 50:] - source_feature[:, 50:]), 1)

            batch_img = []

            for i in range(len(current_batch)):
                image = cv2.imread(self.image_data[current_batch[i][0].int()])
                h, w, c = image.shape
                x1, y1, x2, y2 = int(cut_x1[i] * w), int(cut_y1[i] * h), int(cut_x2[i] * w), int(cut_y2[i] * h)

                source_x1, source_y1, source_x2, source_y2 = int(current_batch[i][4] * w), int(
                    current_batch[i][5] * h), int(current_batch[i][6] * w), int(current_batch[i][7] * h)
                target_x1, target_y1, target_x2, target_y2 = int(current_batch[i][9] * w), int(
                    current_batch[i][10] * h), int(current_batch[i][11] * w), int(current_batch[i][12] * h)

                img = image[y1:y2, x1:x2]
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                source_mask = np.zeros((h, w), dtype=np.float32)
                source_mask[source_y1:source_y2, source_x1:source_x2] = 1  # 1
                target_mask = np.zeros((h, w), dtype=np.float32)
                target_mask[target_y1:target_y2, target_x1:target_x2] = 1  # 1

                source_mask = source_mask[y1:y2, x1:x2]
                target_mask = target_mask[y1:y2, x1:x2]

                source_mask = cv2.resize(source_mask, (224, 224), interpolation=cv2.INTER_AREA)
                target_mask = cv2.resize(target_mask, (224, 224), interpolation=cv2.INTER_AREA)

                img = transforms.ToPILImage()(img)
                img = transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0)(img)


                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                batch_img.append(torch.cat((source_mask, img, target_mask), 0))
            batch_img = torch.stack(batch_img)

            # neg samples
            img_id = np.random.randint(0, len(self.relation_labels), size=self.group)
            for img_i in range(len(img_id)):
                l1 = torch.from_numpy(self.object_labels[img_id[img_i]])
                l2 = torch.from_numpy(self.relation_labels[img_id[img_i]])
                obj_num = len(l1)
                # nosiy_location = torch.randn((obj_num, 4), out=None) * 0.1
                # l1[:, 1:] = l1[:, 1:] + nosiy_location
                source_id = np.random.randint(0, obj_num, size=self.sample_num)
                target_id = np.random.randint(0, obj_num, size=self.sample_num)
                s_t = np.concatenate((source_id.reshape(-1, 1), target_id.reshape(-1, 1)), 1)
                no_relation_index = []
                for j in range(len(s_t)):
                    if s_t[j].tolist() not in l2[:, :2].tolist():
                        no_relation_index.append(s_t[j])
                no_relation_index = torch.from_numpy(np.array(no_relation_index))

                image = cv2.imread(self.image_data[img_id[img_i]])
                h, w, c = image.shape
                x1, y1, x2, y2 = l1[:, 1] * w, l1[:, 2] * h, l1[:, 3] * w, l1[:, 4] * h

                # [word_vector,c_x,c_y,x1,y1,x2,y2]
                obj_location_feature = torch.cat(
                    (((l1[:, 3] - l1[:, 1]) / 2).unsqueeze(1), ((l1[:, 4] - l1[:, 2]) / 2).unsqueeze(1), l1[:, 1:]), 1)
                obj_word_feature = self.word_vec_list[l1[:, 0].long()]

                # print(obj_location_feature.repeat((1,5)).size())
                obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)

                if len(no_relation_index) > 1:
                    no_rel_feature = torch.cat(
                        (obj[no_relation_index[:, 0].long()], obj[no_relation_index[:, 1].long()],
                         obj[no_relation_index[:, 1].long()][:, 50:] - obj[no_relation_index[:, 0].long()][:, 50:]), 1)
                    no_rel_label = torch.cat((torch.tensor([0 for i in range(len(no_relation_index))]).unsqueeze(1),
                                              torch.tensor([0 for i in range(len(no_relation_index))]).unsqueeze(1)), 1)
                    no_relation_batch_img = []
                    for _r in range(len(no_relation_index)):
                        source_x1, source_y1, source_x2, source_y2 = int(x1[no_relation_index[_r][0]]), int(
                            y1[no_relation_index[_r][0]]), int(x2[no_relation_index[_r][0]]), int(
                            y2[no_relation_index[_r][0]])
                        target_x1, target_y1, target_x2, target_y2 = int(x1[no_relation_index[_r][1]]), int(
                            y1[no_relation_index[_r][1]]), int(x2[no_relation_index[_r][1]]), int(
                            y2[no_relation_index[_r][1]])

                        cut_x1 = min((source_x1, target_x1))
                        cut_y1 = min((source_y1, target_y1))
                        cut_x2 = max((source_x2, target_x2))
                        cut_y2 = max((source_y2, target_y2))

                        cut_x1 = torch.clip(cut_x1 - torch.rand(1) * 0.1, 0, w).int()
                        cut_y1 = torch.clip(cut_y1 - torch.rand(1) * 0.1, 0, h).int()
                        cut_x2 = torch.clip(cut_x2 + torch.rand(1) * 0.1, 0, w).int()
                        cut_y2 = torch.clip(cut_y2 + torch.rand(1) * 0.1, 0, h).int()

                        img = image[cut_y1:cut_y2, cut_x1:cut_x2]
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                        source_mask = np.zeros((h, w), dtype=np.float32)
                        source_mask[source_y1:source_y2, source_x1:source_x2] = 1  # 1
                        target_mask = np.zeros((h, w), dtype=np.float32)
                        target_mask[target_y1:target_y2, target_x1:target_x2] = 1  # 1

                        source_mask = source_mask[cut_y1:cut_y2, cut_x1:cut_x2]
                        target_mask = target_mask[cut_y1:cut_y2, cut_x1:cut_x2]

                        source_mask = cv2.resize(source_mask, (224, 224), interpolation=cv2.INTER_AREA)
                        target_mask = cv2.resize(target_mask, (224, 224), interpolation=cv2.INTER_AREA)

                        img = transforms.ToPILImage()(img)
                        img = transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0)(img)


                        img = transforms.ToTensor()(img)
                        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                        source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                        target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                        no_relation_batch_img.append(torch.cat((source_mask, img, target_mask), 0))
                    no_relation_batch_img = torch.stack(no_relation_batch_img)

                    batch_img = torch.cat((batch_img, no_relation_batch_img), 0)
                    batch_word_feature = torch.cat((batch_word_feature, no_rel_feature), 0)
                    batch_label = torch.cat((batch_label, no_rel_label), 0)

            return batch_img, batch_word_feature, batch_label


        else:
            current_batch = self.object_pair_labels[self.object_pair_labels[:, 0] == index]
            source_word_feature = self.word_vec_list[current_batch[:, 3].long()]
            source_bbox = current_batch[:, 4:8]
            source_bbox = torch.cat((((source_bbox[:, 2] - source_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((source_bbox[:, 3] - source_bbox[:, 1]) / 2).unsqueeze(1), source_bbox), 1)
            traget_word_feature = self.word_vec_list[current_batch[:, 8].long()]
            traget_bbox = current_batch[:, 9:13]
            traget_bbox = torch.cat((((traget_bbox[:, 2] - traget_bbox[:, 0]) / 2).unsqueeze(1),
                                     ((traget_bbox[:, 3] - traget_bbox[:, 1]) / 2).unsqueeze(1), traget_bbox), 1)
            source_feature = torch.cat((source_word_feature, source_bbox.repeat((1, 5))), 1)
            target_feature = torch.cat((traget_word_feature, traget_bbox.repeat((1, 5))), 1)
            word_feature = torch.cat((source_feature, target_feature, target_feature[:, 50:] - source_feature[:, 50:]),
                                     1)

            cut_x1, _ = torch.min(current_batch[:, [4, 9]], dim=1)
            cut_y1, _ = torch.min(current_batch[:, [5, 10]], dim=1)
            cut_x2, _ = torch.max(current_batch[:, [6, 11]], dim=1)
            cut_y2, _ = torch.max(current_batch[:, [7, 12]], dim=1)
            image = cv2.imread(self.image_data[index])
            h, w, c = image.shape
            batch_img = []
            for i in range(len(current_batch)):
                x1, y1, x2, y2 = int(cut_x1[i] * w), int(cut_y1[i] * h), int(cut_x2[i] * w), int(cut_y2[i] * h)
                source_x1, source_y1, source_x2, source_y2 = int(current_batch[i][4] * w), int(
                    current_batch[i][5] * h), int(current_batch[i][6] * w), int(current_batch[i][7] * h)
                target_x1, target_y1, target_x2, target_y2 = int(current_batch[i][9] * w), int(
                    current_batch[i][10] * h), int(current_batch[i][11] * w), int(current_batch[i][12] * h)

                img = image[y1:y2, x1:x2]
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                source_mask = np.zeros((h, w), dtype=np.float32)
                source_mask[source_y1:source_y2, source_x1:source_x2] = 1  # 1
                target_mask = np.zeros((h, w), dtype=np.float32)
                target_mask[target_y1:target_y2, target_x1:target_x2] = 1  # 1
                source_mask = source_mask[y1:y2, x1:x2]
                target_mask = target_mask[y1:y2, x1:x2]
                source_mask = cv2.resize(source_mask, (224, 224), interpolation=cv2.INTER_AREA)
                target_mask = cv2.resize(target_mask, (224, 224), interpolation=cv2.INTER_AREA)

                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                batch_img.append(torch.cat((source_mask, img, target_mask), 0))
            batch_img = torch.stack(batch_img)
            l1 = torch.from_numpy(self.object_labels[index])
            l2 = torch.from_numpy(self.relation_labels[index])

            return batch_img, word_feature, l1, l2, self.image_data[index]

    def __len__(self):
        if self.training == True:
            return len(self.object_pair_labels) // (self.group * self.sample_num)
        else:
            return len(self.relation_labels)
