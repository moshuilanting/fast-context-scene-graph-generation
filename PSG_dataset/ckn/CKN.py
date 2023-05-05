#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：622
@File ：model.py
@Author ：jintianlei
@Date : 2022/6/22
"""
import torch
import torch.nn as nn


class FC_Net(nn.Module):
    def __init__(self,dataset_name='VG'):
        super().__init__()
        self.dataset_name = dataset_name
        self.fc1 = nn.Linear(190, 256)
        self.fc2 = nn.Linear(256, 512)
        if self.dataset_name=='PSG':
            self.fc3 = nn.Linear(512, 57)
        else:
            self.fc3 = nn.Linear(512, 51)

    def forward(self, x):

        y = nn.functional.relu(self.fc1(x))
        y = nn.functional.relu(self.fc2(y))
        y = self.fc3(y)

        return y


class FC_Net_Parallel(nn.Module):
    def __init__(self,dataset_name='VG'):
        super().__init__()
        self.dataset_name = dataset_name
        self.word_fc1 = nn.Linear(100, 256)
        self.word_fc2 = nn.Linear(256, 256)

        self.position_fc1 = nn.Linear(90, 256)
        self.position_fc2 = nn.Linear(256, 256)

        self.cat_fc = nn.Linear(512, 512)

        if self.dataset_name=='PSG':
            self.fc3 = nn.Linear(512, 57)
        else:
            self.fc3 = nn.Linear(512, 51)

    def forward(self, x):
        #x [:50] wordvector 1 ; x[50:80] posititionvector 1 ; x[80:130] wordvector 2 ; x[130:160] positionvector 2;  x[160:190] delta position

        wordvector = torch.cat((x[:,:50], x[:,80:130]),dim=1)
        positionvector = torch.cat((x[:,50:80], x[:,130:160], x[:,160:]),dim=1)


        wordvector = nn.functional.relu(self.word_fc1(wordvector))
        wordvector = nn.functional.relu(self.word_fc2(wordvector))

        positionvector = nn.functional.relu(self.position_fc1(positionvector))
        positionvector = nn.functional.relu(self.position_fc2(positionvector))

        y = torch.cat((wordvector, positionvector), dim=1)

        y = nn.functional.relu(self.cat_fc(y))
        y = self.fc3(y)

        return y

if __name__=="__main__":
    import numpy as np
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    model = FC_Net()

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
