from dataset import Scene_Graph
import torch
from ckn.CKN import FC_Net
from loss import SGG_ComputeLoss
import os
import shutil
from datetime import datetime
from datapath import train_object_label_dir,train_relation_label_dir,depth_label_dir

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epochs = 2500
    lr = 0.001
    sampling = 'EPBS'  #EBS or PBS or EPBS

    device = torch.device('cuda', 0)
    model = FC_Net('VG')
    model.to(device)

    # object_label_dir = train_object_label_dir # all .txt in train package
    # relation_label_dir = train_relation_label_dir
    compute_loss = SGG_ComputeLoss(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # data = Scene_Graph(object_label_dir, relation_label_dir,True)

    data = Scene_Graph(train_object_label_dir, train_relation_label_dir,depth_label_dir, group=20, sample_num=10, sampling=sampling,training=True)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=1, # batch_size == 1
                                               shuffle=True,
                                               num_workers=8)

    max_steps = len(train_loader)

    log_dir='runs'
    logdir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    min_loss = 1
    lr_decay_flag = 0
    for epoch in range(epochs):
        total_loss = 0
        total_rel_loss = 0
        total_conf_loss = 0
        total_cost_time = torch.tensor([0.0,0.0,0.0])
        for batch, (relations_feature,relations) in enumerate(train_loader):
            relations_feature = relations_feature[0].to(device)
            relations = relations[0].to(device)

            pred_relations = model(relations_feature)
            rel_loss,conf_loss = compute_loss(pred_relations,relations)
            loss = rel_loss + conf_loss

            total_rel_loss += rel_loss
            total_conf_loss += conf_loss
            total_loss += loss

            loss.backward()  # loss accumulation
            optimizer.step()
            optimizer.zero_grad()


        print('epoch:{:d}/{:d} batch:{:d}/{:d} mean_total_loss:{:6f} mean_rel_loss:{:6f} mean_conf_loss:{:6f} lr:{:6f}'.format(epoch, epochs, (batch + 1),
                                                                                           max_steps,
                                                                                           total_loss / (batch + 1),total_rel_loss / (batch + 1),total_conf_loss / (batch + 1),
                                                                                           optimizer.state_dict()['param_groups'][0]['lr']))



        if (total_loss / (batch + 1)) < min_loss:
            min_loss = (total_loss/ (batch + 1))
            lr_decay_flag = 0
        else:
            lr_decay_flag += 1

        if lr_decay_flag>30:
            scheduler.step()
            lr_decay_flag = 0

        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, os.path.join(logdir, 'last.pt'))

        if (epoch+1) % 1000 == 0:
            torch.save(checkpoint, os.path.join(logdir, '{:d}_last.pt'.format(epoch)))
