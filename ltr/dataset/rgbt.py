import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings

class RGBT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        self.root = env_settings().rgbt_dir if root is None else root
        super().__init__('RGBT', root, image_loader)
        self.split = split
        # video_name for each sequence
        if split=='train': # RGBT234
            self.sequence_list = ['afterrain', 'aftertree', 'baby', 'baginhand', 'baketballwaliking', 'balancebike', 'basketball2', 'bicyclecity', 'bike', 'bikeman', 'bikemove1', 'biketwo', 'blackwoman', 'bluebike', 'blueCar', 'boundaryandfast', 'bus6', 'call', 'car', 'car20', 'car3', 'car37', 'car4', 'car41', 'car66', 'caraftertree', 'carLight', 'carnotfar', 'carnotmove', 'carred', 'child', 'child1', 'child3', 'child4', 'children2', 'children3', 'children4', 'crossroad', 'crouch', 'cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'diamond', 'dog', 'dog1', 'dog10', 'dog11', 'elecbike2', 'elecbike3', 'elecbikechange2', 'elecbikeinfrontcar', 'elecbikewithhat', 'face1', 'floor-1', 'flower1', 'flower2', 'fog', 'fog6', 'glass', 'glass2', 'graycar2', 'green', 'greentruck', 'greyman', 'greywoman', 'guidepost', 'hotglass', 'hotkettle', 'inglassandmobile', 'jump', 'kettle', 'kite2', 'kite4', 'luggage', 'man2', 'man22', 'man23', 'man24', 'man26', 'man28', 'man29', 'man3', 'man4', 'man45', 'man5', 'man55', 'man68', 'man69', 'man7', 'man8', 'man88', 'man9', 'manafterrain', 'mancross1', 'mancrossandup', 'mandrivecar', 'manfaraway', 'maninblack', 'maninglass', 'maningreen2', 'maninred', 'manoccpart', 'manonboundary', 'manonelecbike', 'manontricycle', 'manout2', 'manup', 'manwithbag', 'manwithbag4', 'manwithbasketball', 'manwithluggage', 'manwithumbrella', 'manypeople', 'manypeople1', 'manypeople2', 'mobile', 'night2', 'nightrun', 'nightthreepeople', 'notmove', 'oldman', 'oldman2', 'oldwoman', 'orangeman1', 'people', 'people1', 'people3', 'playsoccer', 'push', 'rainingwaliking', 'redbag', 'redcar', 'redcar2', 'redmanchange', 'rmo', 'run', 'run1', 'run2', 'scooter', 'shake', 'shoeslight', 'single1', 'single3', 'soccer', 'soccer2', 'soccerinhand', 'straw', 'stroller', 'supbus', 'supbus2', 'takeout', 'tallman', 'threeman', 'threeman2', 'threepeople', 'threewoman2', 'together', 'toy1', 'toy3', 'toy4', 'tree2', 'tree3', 'tree5', 'trees', 'tricycle', 'tricycle1', 'tricycle2', 'tricycle6', 'tricycle9', 'tricyclefaraway', 'tricycletwo', 'twoelecbike', 'twoelecbike1', 'twoman', 'twoman1', 'twoman2', 'twoperson', 'twowoman', 'twowoman1', 'walking40', 'walking41', 'walkingman', 'walkingman1', 'walkingman12', 'walkingman20', 'walkingman41', 'walkingmantiny', 'walkingnight', 'walkingtogether', 'walkingtogether1', 'walkingtogetherright', 'walkingwithbag1', 'walkingwithbag2', 'walkingwoman', 'whitebag', 'whitecar', 'whitecar3', 'whitecar4', 'whiteman1', 'whitesuv', 'woamn46', 'woamnwithbike', 'woman', 'woman1', 'woman100', 'woman2', 'woman3', 'woman4', 'woman48', 'woman6', 'woman89', 'woman96', 'woman99', 'womancross', 'womanfaraway', 'womaninblackwithbike', 'womanleft', 'womanpink', 'womanred', 'womanrun', 'womanwithbag6', 'yellowcar']
        elif split=='test': # GTOT
            self.sequence_list = ['BlackCar', 'BlackSwan1', 'BlueCar', 'BusScale', 'BusScale1', 'Crossing', 'crowdNig', 'Cycling', 'DarkNig', 'Exposure2', 'Exposure4', 'fastCar2', 'FastCarNig', 'FastMotor', 'FastMotorNig', 'Football', 'GarageHover', 'Gathering', 'GoTogether', 'Jogging', 'Minibus', 'Minibus1', 'MinibusNig', 'Motorbike', 'Motorbike1', 'MotorNig', 'occBike', 'OccCar-1', 'OccCar-2', 'Otcbvs', 'Otcbvs1', 'Pool', 'Quarreling', 'RainyCar1', 'RainyCar2', 'RainyMotor1', 'RainyMotor2', 'RainyPeople', 'Running', 'Torabi', 'Torabi1', 'Tricycle', 'tunnel', 'Walking', 'WalkingNig', 'WalkingNig1', 'WalkingOcc']

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'rgbt'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',' if self.split=='train' else '\t', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        # seq_name = str(seq_name)
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img'))])[frame_id])
        return self.image_loader(frame_path)

    def get_frames(self, seq_name, frame_ids, anno=None):
        # seq_name = str(seq_name)
        # import pdb
        # pdb.set_trace()
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta
