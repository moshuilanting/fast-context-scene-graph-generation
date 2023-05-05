#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：SG_OID
@File ：oid_infer.py
@Author ：jintianlei
@Date : 2023/2/6
"""
import os
import torch
from ckn.CKN import FC_Net
import torchtext
from tqdm import tqdm
import pickle

names = ['no','Tortoise', 'Container', 'Magpie', 'Turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ', 'Deck', 'Apple', 'Eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Beard', 'Bird', 'Meter', 'Traffic', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washer', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Food', 'Shorts', 'Snack', 'Bus', 'Boy', 'Screwdriver', 'Wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Bear', 'Woodpecker', 'Jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Arch', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Registration', 'Lantern', 'Toaster', 'Light', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Keyboard', 'Printer', 'Traffic', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Hydrant', 'Vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Equipment', 'Cattle', 'Cello', 'Motorboat', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Mouse', 'Cookie', 'Building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Monitor', 'Box', 'Stapler', 'Christmas', 'Hat', 'Equipment', 'Couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Mouth', 'Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet', 'Couch', 'Cricket', 'Melon', 'Spatula', 'Whiteboard', 'Sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Can', 'Mug', 'Tap', 'Seal', 'Stretcher', 'Opener', 'Goggles', 'Body', 'Roller', 'Cup', 'Board', 'Blender', 'Fixture', 'Sign', 'Things', 'Volleyball', 'Vase', 'Cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Towel', 'Products', 'Food', 'Beret', 'Treehouse', 'Frisbee', 'Skirt', 'Stove', 'Shakers', 'Fan', 'Powder', 'Fax', 'Fruit', 'Chips', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'Horn', 'Blind', 'Foot', 'Vehicle', 'Jacket', 'Egg', 'Light', 'Guitar', 'Pillow', 'Leg', 'Isopod', 'Grape', 'Ear', 'Sockets', 'Panda', 'Giraffe', 'Woman', 'Handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Stick', 'Glove', 'Mixer', 'Marine', 'Utensil', 'Switch', 'House', 'Horse', 'Stationary', 'Hammer', 'Fan', 'Sofa', 'Adhesive', 'Harp', 'Sandal', 'Helmet', 'Saucer', 'Harpsichord', 'Hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Straw', 'Insect', 'Dryer', 'Kitchenware', 'Rower', 'Invertebrate', 'Processor', 'Bookcase', 'Refrigerator', 'Stove', 'Punch', 'Fig', 'Shakers', 'Jaguar', 'Golf', 'Accessory', 'Clock', 'Cabinetry', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Opener', 'Lynx', 'Lavender', 'Lighthouse', 'Dumbbell', 'Head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard', 'Mammal', 'Mouse', 'Motorcycle', 'Instrument', 'Cap', 'Pan', 'Snowplow', 'Cabinet', 'Missile', 'Bust', 'Man', 'Iron', 'Milk', 'Binder', 'Plate', 'iphone', 'Things', 'Mushroom', 'Crutch', 'Pitcher', 'Mirror', 'Device', 'Bat', 'Case', 'Keyboard', 'Scoreboard', 'Briefcase', 'Chopper', 'Nail', 'Tennis', 'Bag', 'Oboe', 'Drawer', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Spray', 'Equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Crib', 'Manatee', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'Heels', 'Panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Sandwich', 'Snowboard', 'Sword', 'Frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Bench', 'Snake', 'Place', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Basket', 'Spray', 'Trousers', 'Bowling', 'Helm', 'Truck', 'Cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Products', 'Jug', 'Cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Tissue', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Goblet', 'Countertop', 'Computer', 'Container', 'Pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Dryer', 'Dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Flagstaff', 'Store', 'Bomb', 'Bench', 'Icecream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Butterfly', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash', 'Racket', 'Face', 'Arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Sunflower', 'Microwave', 'Honeycomb', 'Marine', 'Sealion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Rack', 'Light', 'Telephone', 'Uniform', 'Racquet', 'Clock', 'Tray', 'Dining', 'Kennel', 'Stand', 'Shelf', 'Accessory', 'Tissue', 'Cooker', 'Appliance', 'Tire', 'Ruler', 'Bag', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band', 'Animal', 'Pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Hand', 'Skunk', 'Teddy', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Beam', 'Sandwich', 'Shrimp', 'Sewing', 'Binoculars', 'Skate', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remoto', 'Wheelchair', 'Rugby', 'Armadillo', 'Maracas', 'Helmet']


class ckn_inference():
    def __init__(self, weights, device):

        self.device = torch.device('cuda:' + device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = FC_Net('OID')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(weights, map_location=self.device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
        word_vec_list = []
        for name_index in range(len(names)):
            word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
            word_vec_list.append(word_vec)
        self.word_vec_list = torch.stack(word_vec_list).to(self.device)

    def predict(self, boxlists):
        for i in range(len(boxlists)):
            pred_scores = boxlists[i].extra_fields['pred_scores']

            boxlists[i].bbox = boxlists[i].bbox[torch.argsort(pred_scores, descending=True)]
            boxlists[i].extra_fields['pred_labels'] = boxlists[i].extra_fields['pred_labels'][
                torch.argsort(pred_scores, descending=True)]
            boxlists[i].extra_fields['pred_scores'] = boxlists[i].extra_fields['pred_scores'][
                torch.argsort(pred_scores, descending=True)]

            bboxs = boxlists[i].bbox
            pred_labels = boxlists[i].extra_fields['pred_labels']
            pred_scores = boxlists[i].extra_fields['pred_scores']

            width = boxlists[i].size[0]
            height = boxlists[i].size[1]


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
            obj_pair_confidence = torch.cat(
                (predited_obj_confidence[obj_pair[:, 0].long()], predited_obj_confidence[obj_pair[:, 1].long()]), 1)

            pred_relations = self.model(rel_feature)
            pred_relations = torch.sigmoid(pred_relations)
            pred_relations_conf, pred_relations = pred_relations[:, 0:1], pred_relations[:, 1:]

            pred_relations = torch.cat(
                [torch.tensor([[0] for i in range(len(pred_relations))], device=self.device), pred_relations], dim=1)

            pred_relations = pred_relations * pred_relations_conf
            relation_max, relation_argmax = torch.max(pred_relations, dim=1)

            obj_pair_confidence = torch.cat((obj_pair_confidence, relation_max.unsqueeze(1)), 1)
            obj_pair_confidence_score = obj_pair_confidence[:, 0] * obj_pair_confidence[:, 1] * obj_pair_confidence[:,
                                                                                                2]

            head_semantic = torch.cat(
                [obj_pair[:, :2].to(self.device), relation_argmax.view(-1, 1), relation_max.view(-1, 1),
                 pred_relations], dim=1)

            head_semantic = head_semantic[torch.argsort(obj_pair_confidence_score, descending=True)][:4096]
            obj_pair_confidence = obj_pair_confidence[torch.argsort(obj_pair_confidence_score, descending=True)][:4096]

            boxlists[i].extra_fields['rel_pair_idxs'] = head_semantic[:, :2]
            boxlists[i].extra_fields['pred_rel_scores'] = head_semantic[:, 4:]
            boxlists[i].extra_fields['pred_rel_labels'] = head_semantic[:, 2]
            boxlists[i].extra_fields['relness'] = obj_pair_confidence


        return boxlists

if __name__=="__main__":

    ckn = ckn_inference(weights='ckn/ckn_oid_PBS.pt',device='0')
    save_bboxlist = []

    f = open("openimage_v6_test/bbox_save.bin", 'rb')
    bboxlists = pickle.load(f)
    f.close()


    results_dict = []
    with torch.no_grad():
        for bboxlist in tqdm(bboxlists,total=len(bboxlists)):
            output = ckn.predict(bboxlist)
            results_dict.append(output[0])

    eval_results = torch.load('openimage_v6_test/eval_results_backup.pytorch', map_location=torch.device("cpu"))
    eval_results['predictions'] = results_dict
    torch.save(eval_results,'openimage_v6_test/eval_results.pytorch')
