#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SG_OID
@File ：oid_infer.py
@Author ：jintianlei
@Date : 2023/3/6
"""
import copy
import json
import os
import torch
from ckn.CKN import FC_Net
from vdn.VDN import visual_rel_model
import torchtext
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import torchvision.transforms as transforms

names = ['no','Tortoise', 'Container', 'Magpie', 'Turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ', 'Deck', 'Apple', 'Eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Beard', 'Bird', 'Meter', 'Traffic', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washer', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Food', 'Shorts', 'Snack', 'Bus', 'Boy', 'Screwdriver', 'Wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Bear', 'Woodpecker', 'Jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Arch', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Registration', 'Lantern', 'Toaster', 'Light', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Keyboard', 'Printer', 'Traffic', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Hydrant', 'Vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Equipment', 'Cattle', 'Cello', 'Motorboat', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Mouse', 'Cookie', 'Building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Monitor', 'Box', 'Stapler', 'Christmas', 'Hat', 'Equipment', 'Couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Mouth', 'Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet', 'Couch', 'Cricket', 'Melon', 'Spatula', 'Whiteboard', 'Sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Can', 'Mug', 'Tap', 'Seal', 'Stretcher', 'Opener', 'Goggles', 'Body', 'Roller', 'Cup', 'Board', 'Blender', 'Fixture', 'Sign', 'Things', 'Volleyball', 'Vase', 'Cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Towel', 'Products', 'Food', 'Beret', 'Treehouse', 'Frisbee', 'Skirt', 'Stove', 'Shakers', 'Fan', 'Powder', 'Fax', 'Fruit', 'Chips', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'Horn', 'Blind', 'Foot', 'Vehicle', 'Jacket', 'Egg', 'Light', 'Guitar', 'Pillow', 'Leg', 'Isopod', 'Grape', 'Ear', 'Sockets', 'Panda', 'Giraffe', 'Woman', 'Handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Stick', 'Glove', 'Mixer', 'Marine', 'Utensil', 'Switch', 'House', 'Horse', 'Stationary', 'Hammer', 'Fan', 'Sofa', 'Adhesive', 'Harp', 'Sandal', 'Helmet', 'Saucer', 'Harpsichord', 'Hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Straw', 'Insect', 'Dryer', 'Kitchenware', 'Rower', 'Invertebrate', 'Processor', 'Bookcase', 'Refrigerator', 'Stove', 'Punch', 'Fig', 'Shakers', 'Jaguar', 'Golf', 'Accessory', 'Clock', 'Cabinetry', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Opener', 'Lynx', 'Lavender', 'Lighthouse', 'Dumbbell', 'Head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard', 'Mammal', 'Mouse', 'Motorcycle', 'Instrument', 'Cap', 'Pan', 'Snowplow', 'Cabinet', 'Missile', 'Bust', 'Man', 'Iron', 'Milk', 'Binder', 'Plate', 'iphone', 'Things', 'Mushroom', 'Crutch', 'Pitcher', 'Mirror', 'Device', 'Bat', 'Case', 'Keyboard', 'Scoreboard', 'Briefcase', 'Chopper', 'Nail', 'Tennis', 'Bag', 'Oboe', 'Drawer', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Spray', 'Equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Crib', 'Manatee', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'Heels', 'Panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Sandwich', 'Snowboard', 'Sword', 'Frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Bench', 'Snake', 'Place', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Basket', 'Spray', 'Trousers', 'Bowling', 'Helm', 'Truck', 'Cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Products', 'Jug', 'Cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Tissue', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Goblet', 'Countertop', 'Computer', 'Container', 'Pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Dryer', 'Dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Flagstaff', 'Store', 'Bomb', 'Bench', 'Icecream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Butterfly', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash', 'Racket', 'Face', 'Arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Sunflower', 'Microwave', 'Honeycomb', 'Marine', 'Sealion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Rack', 'Light', 'Telephone', 'Uniform', 'Racquet', 'Clock', 'Tray', 'Dining', 'Kennel', 'Stand', 'Shelf', 'Accessory', 'Tissue', 'Cooker', 'Appliance', 'Tire', 'Ruler', 'Bag', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band', 'Animal', 'Pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Hand', 'Skunk', 'Teddy', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Beam', 'Sandwich', 'Shrimp', 'Sewing', 'Binoculars', 'Skate', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remoto', 'Wheelchair', 'Rugby', 'Armadillo', 'Maracas', 'Helmet']


# def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None, logger=None):
#     """
#
#     :param model:
#     :param data_loader:
#     :param device:
#     :param synchronize_gather:  gather the predictions during the training,
#                                 rather than gathering all predictions after the training
#     :param timer:
#     :return:
#     """
#     model.eval()
#     results_dict = {}
#     cpu_device = torch.device("cpu")
#     for _, batch in enumerate(bboxlists):
#         with torch.no_grad():
#             images, targets, image_ids = batch
#             targets = [target.to(device) for target in targets]
#             if timer:
#                 timer.tic()
#             if cfg.TEST.BBOX_AUG.ENABLED:
#                 output = im_detect_bbox_aug(model, images, device)
#             elif len(bboxlists)!=0:
#                 output = ckn.predict(bboxlists[_])
#             else:
#                 # relation detection needs the targets
#                 output = model(images.to(device), targets, logger=logger)
#                 save_bboxlist.append(output)
#                 output = ckn.predict(output)
#             if timer:
#                 if not cfg.MODEL.DEVICE == 'cpu':
#                     torch.cuda.synchronize()
#                 timer.toc()
#             output = [o.to(cpu_device) for o in output]
#
#         else:
#             results_dict.update(
#                 {img_id: result for img_id, result in zip(image_ids, output)}
#             )
#
#     if os.path.exists("save.bin") == False:
#         fBin = open("save.bin", 'ab')
#         pickle.dump(save_bboxlist, fBin)
#         fBin.close()
#
#     return results_dict



class vdn_inference():
    def __init__(self, vdn_weights, ckn_weights,device):

        self.device = torch.device('cuda:' + device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = visual_rel_model('OID')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(vdn_weights, map_location=self.device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)


        self.ckn_model = FC_Net('OID')
        model_dict = self.ckn_model.state_dict()
        pretrained_dict = torch.load(ckn_weights, map_location=self.device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.ckn_model.load_state_dict(model_dict)
        self.ckn_model.to(self.device)

        word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
        word_vec_list = []
        for name_index in range(len(names)):
            word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
            word_vec_list.append(word_vec)
        self.word_vec_list = torch.stack(word_vec_list).to(self.device)

    def predict(self, image_name, boxlists):
        for bboxlist_i in range(len(boxlists)):
            pred_scores = boxlists[bboxlist_i].extra_fields['pred_scores']

            boxlists[bboxlist_i].bbox = boxlists[bboxlist_i].bbox[torch.argsort(pred_scores, descending=True)]
            boxlists[bboxlist_i].extra_fields['pred_labels'] = boxlists[bboxlist_i].extra_fields['pred_labels'][
                torch.argsort(pred_scores, descending=True)]
            boxlists[bboxlist_i].extra_fields['pred_scores'] = boxlists[bboxlist_i].extra_fields['pred_scores'][
                torch.argsort(pred_scores, descending=True)]
            bboxs = boxlists[bboxlist_i].bbox
            current_bbox_hw = copy.copy(bboxs)

            pred_labels = boxlists[bboxlist_i].extra_fields['pred_labels']
            pred_scores = boxlists[bboxlist_i].extra_fields['pred_scores']

            width = boxlists[bboxlist_i].size[0]
            height = boxlists[bboxlist_i].size[1]

            image = cv2.imread(image_name)
            # print(image_name)
            # print(image.shape)

            image = cv2.resize(image,(width,height))

            # for b in range(len(current_bbox_hw)):
            #     cv2.rectangle(image,(int(current_bbox_hw[b][0]),int(current_bbox_hw[b][1])),(int(current_bbox_hw[b][2]),int(current_bbox_hw[b][3])),(255,0,255),2)

            # cv2.imshow('raw',image)
            # cv2.waitKey(1000)
            # print(image.shape)
            # input()
            # print(pred_labels.size())
            # print((bboxs[:,0]/width).unsqueeze(1).size())

            predited_obj = torch.cat((pred_labels.reshape(-1, 1), (bboxs[:, 0] / width).unsqueeze(1),
                                      (bboxs[:, 1] / height).unsqueeze(1), (bboxs[:, 2] / width).unsqueeze(1),
                                      (bboxs[:, 3] / height).unsqueeze(1)), 1)[:80]

            obj_location_feature = torch.cat((((predited_obj[:, 3] - predited_obj[:, 1]) / 2).unsqueeze(1),
                                              ((predited_obj[:, 4] - predited_obj[:, 2]) / 2).unsqueeze(1),
                                              predited_obj[:, 1:]), 1)
            obj_word_feature = self.word_vec_list[predited_obj[:, 0].long()]
            predited_obj_confidence = pred_scores.reshape(-1, 1)
            obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)
            obj_id = torch.arange((len(predited_obj)))
            a_id = obj_id.repeat(len(obj_id), 1).view(-1, )
            b_id = torch.repeat_interleave(obj_id, len(obj_id), dim=0)
            obj_pair = torch.cat((a_id.unsqueeze(1), b_id.unsqueeze(1)), dim=1)
            obj_pair = obj_pair[obj_pair[:, 0] != obj_pair[:, 1]]

            rel_feature = torch.cat((obj[obj_pair[:, 0].long()], obj[obj_pair[:, 1].long()],
                                     obj[obj_pair[:, 1].long()][:, 50:] - obj[obj_pair[:, 0].long()][:, 50:]), 1)
            obj_pair_confidence = torch.cat((predited_obj_confidence[obj_pair[:, 0].long()], predited_obj_confidence[obj_pair[:, 1].long()]), 1)

            pred_relations = self.ckn_model(rel_feature)
            pred_relations = torch.sigmoid(pred_relations)
            pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]

            relation_max, _ = torch.max(pred_relations, dim=1)

            #print(obj_pair_confidence)
            a, indices = torch.sort(relation_max.view(-1)*pred_relations_conf.view(-1)*obj_pair_confidence[:,0].view(-1)*obj_pair_confidence[:,1].view(-1), descending=True)

            select_obj_pair = obj_pair[indices[:100]]
            select_rel_feature = rel_feature[indices[:100]]
            pred_relations_conf = pred_relations_conf[indices[:100]]
            pred_relations = pred_relations[indices[:100]]
            obj_pair_confidence = obj_pair_confidence[indices[:100]]
            pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations],dim=1)

            # print(len(select_obj_pair))

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
                    x1, y1, x2, y2 = int(cut_x1[j * 50 + i]), int(cut_y1[j * 50 + i]), int(cut_x2[j * 50 + i]), int(cut_y2[j * 50 + i])

                    source_x1, source_y1, source_x2, source_y2 = int(current_batch_visual_bbox_pair[i][0]), int(current_batch_visual_bbox_pair[i][1]), int(current_batch_visual_bbox_pair[i][2]), int(current_batch_visual_bbox_pair[i][3])
                    target_x1, target_y1, target_x2, target_y2 = int(current_batch_visual_bbox_pair[i][4]), int(current_batch_visual_bbox_pair[i][5]), int(current_batch_visual_bbox_pair[i][6]), int(current_batch_visual_bbox_pair[i][7])

                    #print(image.shape,x1,x2,y1,y2)
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

                    # cv2.imshow('1', img)
                    # cv2.imshow('2', source_mask)
                    # cv2.imshow('3', target_mask)
                    # cv2.waitKey(1000)

                    img = transforms.ToTensor()(img)
                    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                    source_mask = torch.from_numpy(source_mask).unsqueeze(0)
                    target_mask = torch.from_numpy(target_mask).unsqueeze(0)
                    batch_img.append(torch.cat((source_mask, img, target_mask), 0))

                batch_img = torch.stack(batch_img).cuda()
                visual_feature, semantic_pred, fusion_relation_pred = self.model(batch_img, batch_word_feature)
                batch_visual_pred.append(fusion_relation_pred)
                batch_semantic_pred.append(semantic_pred)

            pred_fusion_relations = torch.cat(batch_visual_pred, 0)
            pred_semantic_relations = torch.cat(batch_semantic_pred, 0)

            # pred_relations = pred_semantic_relations
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

            # pred_relations = pred_fusion_relations
            _pred_relations_conf, pred_relations = pred_semantic_relations[:, 0:1], pred_fusion_relations[:, 1:]

            pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))]).cuda(), pred_relations],dim=1)
            pred_relations[_pred_relation_indices_neg[:, 0], _pred_relation_indices_neg[:, 1]] = 0.0
            # pred_relations = pred_relations * pred_relations_conf

            if len(pred_relations) > 0:

                pred_relations = pred_relations  #* pred_relations_conf
                relation_max, relation_argmax = torch.max(pred_relations, dim=1)
                # confidence sort
                # head_semantic = torch.cat([obj_pair[:, :2].cuda(), relation_argmax.view(-1, 1), relation_max.view(-1, 1), pred_relations],dim=1)

                obj_pair_confidence = torch.cat((obj_pair_confidence, relation_max.unsqueeze(1)), 1)
                obj_pair_confidence_score = obj_pair_confidence[:, 0] * obj_pair_confidence[:, 1] * obj_pair_confidence[:,2]

                head_semantic = torch.cat([select_obj_pair[:, :2].to(self.device), relation_argmax.view(-1, 1), relation_max.view(-1, 1),pred_relations], dim=1)

                # head_semantic = head_semantic[torch.argsort(head_semantic[:,3],descending=True)][:4096]
                head_semantic = head_semantic[torch.argsort(obj_pair_confidence_score, descending=True)][:4096]
                #print(head_semantic.size())

                obj_pair_confidence = obj_pair_confidence[torch.argsort(obj_pair_confidence_score, descending=True)][:4096]

                boxlists[bboxlist_i].extra_fields['rel_pair_idxs'] = head_semantic[:, :2]
                boxlists[bboxlist_i].extra_fields['pred_rel_scores'] = head_semantic[:, 4:]
                boxlists[bboxlist_i].extra_fields['pred_rel_labels'] = head_semantic[:, 2]
                boxlists[bboxlist_i].extra_fields['relness'] = obj_pair_confidence

        return boxlists


if __name__=="__main__":

    yourpath = '/media/jtl/615abae4-9da3-4fa7-8e7b-9e9925e51311/pysgg/datasets'

    infer = vdn_inference(vdn_weights='vdn/EPBS2023-03-10_09-15-18_vdn/last.pt',ckn_weights = 'ckn/ckn_oid_EPBS_n.pt',device='0')
    save_bboxlist = []

    f = open("openimage_v6_test/bbox_save.bin", 'rb')
    bboxlists = pickle.load(f)
    f.close()


    with open(os.path.join(yourpath,'openimages/open-imagev6/annotations/vrd-test-anno.json'),'r') as infile:
        raw_data = json.load(infile)

    results_dict = []
    with torch.no_grad():
        for data_info,bboxlist in tqdm(zip(raw_data,bboxlists),total=len(bboxlists)):
            image_path = os.path.join(yourpath,'openimages/open-imagev6/images',data_info["img_fn"])+'.jpg'
            output = infer.predict(image_path,bboxlist)
            results_dict.append(output[0])

    # print(len(bboxlists),len(raw_data))
    # for i in range(len(bboxlists[0])):
    #     print(bboxlists[0])
    #     print(bboxlists[0][i].bbox,bboxlists[0][i].get_field('pred_labels'))
    # print(raw_data[0]['bbox'],raw_data[0]['det_labels'])
    # input()
    # results_dict = []
    # with torch.no_grad():
    #     for bboxlist in tqdm(bboxlists,total=len(bboxlists)):
    #         #print(bboxlist)
    #         #images, targets, image_ids = batch
    #         output = ckn.predict(bboxlist)
    #         results_dict.append(output[0])
    #
    eval_results = torch.load('openimage_v6_test/eval_results_backup.pytorch', map_location=torch.device("cpu"))
    eval_results['predictions'] = results_dict
    torch.save(eval_results,'openimage_v6_test/eval_results.pytorch')
