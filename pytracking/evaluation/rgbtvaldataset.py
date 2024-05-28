import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os

class RGBTVALDataset(BaseDataset):
    
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])


    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=['', '\t', ','], dtype=np.float64)
        img_list = sorted([p for p in os.listdir(os.path.join(sequence_path, 'img'))])
        frames = [os.path.join(sequence_path, 'img', img) for img in img_list]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_info['name'], frames, 'rgbtval', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_list(self):
        sequence_list = ['BlackCar', 'BlueCar', 'BusScale', 'BusScale1', 'Crossing', 'crowdNig', 'Cycling', 'DarkNig', 'Exposure2', 'Exposure4', 'FastCarNig', 'FastMotor', 'FastMotorNig', 'Football', 'GarageHover', 'Gathering', 'GoTogether', 'Jogging', 'Minibus', 'Minibus1', 'MinibusNig', 'Motorbike', 'Motorbike1', 'MotorNig', 'occBike', 'OccCar-1', 'OccCar-2', 'Otcbvs', 'Otcbvs1', 'Pool', 'Quarreling', 'RainyCar1', 'RainyMotor1', 'RainyMotor2', 'RainyPeople', 'Running', 'Torabi', 'Torabi1', 'Tricycle', 'tunnel', 'Walking', 'WalkingNig', 'WalkingOcc']
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = '/home/liulei/zhutianhao/RGBT/'+sequence_info["name"]
            sequence_info["anno_path"] = sequence_info["path"]+'/init.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
