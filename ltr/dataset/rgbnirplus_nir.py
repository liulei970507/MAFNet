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

class RGBNIRPLUS_NIR(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        self.root = env_settings().rgbnirplus_nir_dir if root is None else root
        super().__init__('RGBNIRPLUS_NIR', root, image_loader)

        # video_name for each sequence
        if split=='train':
            self.sequence_list = ['933', '710', '863', '665', '948', '136', '614', '901', '918', '85', '735', '932', '752', '760', '743', '996', '790', '849', '926', '877', '784', '825', '936', '685', '842', '980', '951', '686', '911', '937', '855', '905', '772', '949', '674', '940', '704', '803', '705', '751', '913', '775', '939', '934', '315', '879', '785', '673', '517', '870', '947', '824', '732', '687', '756', '723', '722', '819', '668', '931', '989', '773', '878', '998', '984', '959', '857', '708', '946', '688', '853', '896', '993', '738', '858', '689', '787', '724', '794', '961', '682', '817', '871', '189', '958', '523', '742', '876', '843', '835', '955', '786', '922', '983', '950', '727', '892', '954', '755', '734', '920', '816', '757', '813', '970', '696', '737', '915', '991', '906', '888', '814', '889', '766', '707', '935', '831', '628', '792', '700', '694', '919', '702', '715', '797', '765', '987', '861', '690', '664', '788', '990', '691', '778', '758', '903', '681', '695', '992', '895', '676', '900', '988', '780', '845', '838', '967', '979', '729', '718', '930', '880', '746', '966', '811', '928', '721', '799', '802', '699', '999', '745', '133', '753', '869', '832', '767', '823', '921', '917', '781', '779', '841', '968', '731', '828', '267']
        elif split=='test':
            self.sequence_list = ['963', '972', '800', '666', '862', '716', '791', '962', '840', '897', '697', '711', '774', '994', '899', '846', '798', '726', '730', '834', '827', '754', '768', '805', '851', '975', '985', '667', '1000', '744', '971', '872', '866', '864', '973', '679', '725', '706', '908', '182', '844', '902', '981', '924', '701', '886', '898', '771', '943', '763', '847', '821', '883', '890', '887', '709', '762', '952', '833', '733', '526', '904', '837', '938', '925', '859', '713', '914', '953', '663', '769', '818', '965', '826', '848', 'ba.tar.gz', '703', '698', '806', '670', '829', '874', '186', '907', '852', '777', '885', '873', '684', '875', '712', '795', '770', '969', '720', '789', '960', '929', '945', '964', '812', '776', '717', '793', '749', '719', '894', '747', '759', '881', '974', '736', '750', '850', '728', '190', '675', '977', '739', '891', '882', '997', '804', '672', 'jiu.tar.gz', '830', '822', '923', '782', '944', '978', '692', '868', '809', '810', '909', '986', '801', '808', '976', '807', '942', '910', '740', '941', '856', '693', '714', '748', '957', '860', '865', '680', '671', '677', '796', '867', '912', '927', '995', '854', '836', '916', '815', '764', '683', '884', '761', '669', '678', '783', '893', '820', '982', '839', '741', '956']
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'rgbnir_nir'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img'))])[frame_id])
        # print(frame_path)
        return self.image_loader(frame_path)

    def get_frames(self, seq_name, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]
        # print(frame_ids)
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
