import torch
from vdn.VDN import visual_rel_model
from dataset import Scene_Graph_Image
import os
import shutil
from datetime import datetime
from loss import Visual_Contrast_ComputeLoss
from datapath import image_file, train_object_label_dir, train_relation_label_dir

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epochs = 300
    lr = 0.0002
    sample_group = 4 #4
    sample_number = 5 #5
    sampling = 'EBS' #EBS or PBS or EPBS
    device = torch.device('cuda', 0)
    model = visual_rel_model()
    model_dict = model.state_dict()

    pretrained_dict = torch.load('ckn/CKN+EPBS.pt') # pretrained CKN model
    pretrained_dict = {'image_conv.'+k: v for k, v in pretrained_dict['model'].items() if 'image_conv.'+k in model_dict}
    # for k,v in pretrained_dict.items():
    #     v.requires_grad=False
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    for name,param in model.state_dict(keep_vars=True).items():
        if 'image_conv.fc' in name:
            param.requires_grad = False

    model.to(device)

    image_dir = image_file
    object_label_dir = train_object_label_dir
    relation_label_dir = train_relation_label_dir

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    data = Scene_Graph_Image(image_dir, object_label_dir, relation_label_dir, group= sample_group, sample_num= sample_number,sampling=sampling, training=True)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=8)

    max_steps = len(train_loader)
    log_dir = 'runs'
    logdir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    print(logdir)

    compute_loss = Visual_Contrast_ComputeLoss(device,sample_number)

    min_loss = 100
    lr_decay_flag = 0
    for epoch in range(epochs):
        total_loss = 0

        total_pos_boost_loss = 0
        total_neg_suppress_loss = 0
        total_fusion_conf_loss = 0

        for batch, (batch_img, batch_word_feature,batch_label) in enumerate(train_loader):

            _batch_img = batch_img[0].to(device)
            _batch_label = batch_label[0].to(device)
            _batch_word_feature = batch_word_feature[0].to(device)
            #x,y,fusion
            visual_fusion,y,fusion = model(_batch_img,_batch_word_feature)

            fusion_conf_loss,pos_boost_loss, neg_suppress_loss = compute_loss(visual_fusion,y,fusion,_batch_label)

            loss = fusion_conf_loss+pos_boost_loss+neg_suppress_loss

            loss.backward()  # loss accumulation

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            total_pos_boost_loss += pos_boost_loss.item()
            total_neg_suppress_loss += neg_suppress_loss.item()
            total_fusion_conf_loss += fusion_conf_loss.item()


            if (batch+1) % 10 == 0:
                print('epoch:{:d}/{:d} batch:{:d}/{:d} fusion_conf_loss: {:6f} total_pos_boost_loss: {:6f} total_neg_suppress_loss: {:6f} total_loss:{:6f}  lr:{:6f} lr_decay_flag:{:d}'.format(
                        epoch, epochs, (batch + 1),
                        max_steps,
                        total_fusion_conf_loss / (batch + 1), total_pos_boost_loss / (batch + 1), total_neg_suppress_loss/(batch + 1), total_loss/(batch + 1),
                        optimizer.state_dict()['param_groups'][0]['lr'],lr_decay_flag))

        if (total_loss / (batch + 1)) < min_loss:
            min_loss = (total_loss/ (batch + 1))
            lr_decay_flag = 0
        else:
            lr_decay_flag += 1

        if lr_decay_flag>10:
            scheduler.step()
            lr_decay_flag = 0


        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, os.path.join(logdir, 'last.pt'))

        #compute_loss.save(logdir,'relation_vector.pt')
