#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：623
@File ：image_model.py
@Author ：jintianlei
@Date : 2022/6/23
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=True, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resnet_model(nn.Module):
    def __init__(self, block, layers=[2, 2, 2, 2, 2], dataset_name='VG', baseWidth=26, scale=4):
        self.inplanes = 64
        super(resnet_model, self).__init__()
        #self.baseWidth = baseWidth
        #self.scale = scale

        self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 4, layers[4], stride=1)

        self.fc1 = nn.Linear(190, 256)
        self.fc2 = nn.Linear(256, 512)

        self.visual_embedding_fc1 = nn.Linear(100, 256)
        self.visual_embedding_fc2 = nn.Linear(256, 256)
        self.visual_fc1 = nn.Linear(1040, 512)


        if dataset_name=='PSG':
            self.fc3 = nn.Linear(512, 57)
            self.visual_fc2 = nn.Linear(512, 57)
        else:
            self.fc3 = nn.Linear(512, 51)
            self.visual_fc2 = nn.Linear(512, 51)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, semantic_vector):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        y = nn.functional.relu(self.fc1(semantic_vector))
        y = nn.functional.relu(self.fc2(y))
        y = self.fc3(y)

        word_vector = torch.cat((semantic_vector[:,:50],semantic_vector[:,80:130]),1)

        word_feature = nn.functional.relu(self.visual_embedding_fc1(word_vector))
        word_feature = nn.functional.relu(self.visual_embedding_fc2(word_feature))

        visual_fusion = torch.cat((x.view(-1,784),word_feature),dim=1)
        x = visual_fusion = nn.functional.relu(self.visual_fc1(visual_fusion))

        #fusion = nn.functional.relu(self.fusion_fc3(fusion))
        fusion = self.visual_fc2(x)

        y = torch.sigmoid(y)
        fusion = torch.sigmoid(fusion)
        fusion = 0.3*fusion+0.7*y

        return visual_fusion,y,fusion


class visual_rel_model(nn.Module):
    def __init__(self,dataset_name='VG'):
        super(visual_rel_model, self).__init__()
        self.image_conv = resnet_model(Bottleneck,[2,2,2,2,2],dataset_name)

    def forward(self, concat_image,word_vector):
        #x,y = self.image_conv(concat_image,word_vector)
        visual_fusion,y,fusion_y = self.image_conv(concat_image, word_vector)

        return visual_fusion,y,fusion_y


if __name__=="__main__":
    import numpy as np
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    model = visual_rel_model()

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
