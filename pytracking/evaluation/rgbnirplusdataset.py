import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os

class RGBNIRPLUSDataset(BaseDataset):
    
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbnirplus_path
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
        return Sequence(sequence_info['name'], frames, 'rgbnirplus', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_list(self):
        sequence_list = ['538', '18', '963', '972', '800', '666', '862', '716', '791', '962', '840', '897', '697', '711', '774', '994', '899', '846', '798', '726', '730', '834', '827', '754', '768', '805', '851', '975', '985', '667', '1000', '744', '971', '872', '866', '864', '973', '679', '725', '706', '908', '182', '844', '902', '981', '924', '701', '886', '898', '771', '943', '763', '847', '821', '883', '890', '887', '709', '762', '952', '833', '733', '526', '904', '837', '938', '925', '859', '713', '914', '953', '663', '769', '818', '965', '826', '848', '703', '698', '806', '670', '829', '874', '186', '907', '852', '777', '885', '873', '684', '875', '712', '795', '770', '969', '720', '789', '960', '929', '945', '964', '812', '776', '717', '793', '749', '719', '894', '747', '759', '881', '974', '736', '750', '850', '728', '190', '675', '977', '739', '891', '882', '997', '804', '672', '830', '822', '923', '782', '944', '978', '692', '868', '809', '810', '909', '986', '801', '808', '976', '807', '942', '910', '740', '941', '856', '693', '714', '748', '957', '860', '865', '680', '671', '677', '796', '867', '912', '927', '995', '854', '836', '916', '815', '764', '683', '884', '761', '669', '678', '783', '893', '820', '982', '839', '741', '956']
        
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = self.base_path + sequence_info["name"]
            sequence_info["anno_path"] = sequence_info["path"]+'/'+'groundtruth_rect.txt'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
