import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings

class RGBNIR(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        self.root = env_settings().rgbnir_dir if root is None else root
        super().__init__('RGBNIR', root, image_loader)

        # video_name for each sequence
        if split=='train':
            self.sequence_list = ['1', '3', '4', '5', '9', '10', '11', '13', '14', '15', '16', '17', '23', '24', '25', '27', '30', '34', '35', '36', '37', '39', '42', '46', '48', '49', '52', '53', '54', '56', '62', '63', '65', '66', '68', '71', '72', '73', '74', '75', '77', '78', '79', '82', '83', '86', '87', '88', '89', '91', '93', '94', '95', '97', '102', '104', '105', '106', '110', '111', '113', '117', '121', '123', '125', '126', '130', '134', '138', '144', '146', '148', '149', '152', '154', '155', '156', '158', '161', '164', '165', '168', '169', '170', '171', '172', '173', '175', '177', '178', '180', '181', '183', '184', '185', '187', '193', '194', '196', '199', '200', '201', '205', '207', '208', '212', '213', '215', '216', '217', '218', '219', '221', '224', '225', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '238', '239', '240', '241', '242', '246', '248', '250', '253', '257', '259', '265', '268', '269', '270', '271', '272', '273', '274', '276', '280', '283', '286', '289', '291', '292', '293', '294', '297', '298', '299', '300', '301', '302', '303', '304', '305', '308', '309', '310', '311', '312', '313', '314', '316', '317', '318', '319', '321', '324', '325', '327', '328', '329', '330', '331', '333', '334', '339', '342', '344', '345', '346', '351', '353', '354', '355', '357', '358', '359', '360', '362', '363', '364', '365', '366', '368', '369', '371', '372', '375', '377', '378', '379', '380', '383', '387', '388', '389', '390', '392', '393', '395', '399', '401', '402', '404', '407', '413', '418', '421', '422', '425', '428', '429', '431', '432', '433', '436', '437', '441', '442', '443', '444', '448', '449', '450', '453', '454', '455', '456', '457', '458', '460', '461', '463', '464', '465', '466', '468', '469', '470', '471', '472', '474', '475', '476', '477', '478', '484', '485', '486', '487', '488', '491', '493', '494', '496', '498', '499', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '518', '519', '520', '521', '522', '524', '525', '527', '528', '529', '530', '531', '532', '534', '535', '536', '537', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', '661', '662']
        elif split=='test':
            self.sequence_list = ['2', '6', '7', '8', '12', '19', '20', '21', '22', '26', '28', '29', '31', '32', '33', '38', '40', '41', '43', '44', '45', '47', '50', '51', '55', '57', '58', '59', '60', '61', '64', '67', '69', '70', '76', '80', '81', '84', '90', '92', '96', '98', '99', '100', '101', '103', '107', '108', '109', '112', '114', '115', '116', '118', '119', '120', '122', '124', '127', '128', '129', '131', '132', '135', '137', '139', '140', '141', '142', '143', '145', '147', '150', '151', '153', '157', '159', '160', '162', '163', '166', '167', '174', '176', '179', '188', '191', '192', '195', '197', '198', '202', '203', '204', '206', '209', '210', '211', '214', '220', '222', '223', '226', '237', '243', '244', '245', '247', '249', '251', '252', '254', '255', '256', '258', '260', '262', '263', '264', '266', '275', '277', '278', '279', '281', '282', '284', '285', '287', '288', '290', '295', '296', '306', '307', '320', '322', '323', '326', '332', '335', '336', '337', '338', '340', '341', '343', '347', '348', '349', '350', '352', '356', '361', '367', '370', '373', '374', '376', '381', '382', '384', '385', '386', '391', '394', '396', '397', '398', '400', '403', '405', '406', '408', '409', '410', '411', '412', '414', '415', '416', '417', '419', '420', '423', '424', '426', '427', '430', '434', '435', '438', '439', '440', '445', '446', '447', '451', '452', '459', '462', '467', '473', '479', '480', '481', '482', '483', '489', '490', '492', '495', '497', '500']
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'rgbnir'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)
    
    def _read_modality_anno(self, seq_path):
        gt = np.loadtxt(os.path.join(seq_path, 'modality.tag'))
        return torch.tensor(gt)
    
    def get_sequence_info(self, seq_name):
        # seq_name = str(seq_name)
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        modality = self._read_modality_anno(seq_path)
        valid1 = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid2 = (modality[:] == 0) | (modality[:] == 1) | (modality[:] == 2)
        try:
            valid = valid1 & valid2[:len(valid1)]
        except:
            print('Error seq_path:', seq_path)
            valid = valid1
            
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'modality':modality}
    
    def _get_ms_label(self, seq_path, frame_id):
        rgb_nir_label_file  = np.loadtxt(os.path.join(seq_path, 'modality.tag'))[frame_id]
        return rgb_nir_label_file
    
    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img'))])[frame_id])
        return self.image_loader(frame_path)

    def get_frames(self, seq_name, frame_ids, anno=None):
        # seq_name = str(seq_name)
        # import pdb
        # pdb.set_trace()
        seq_name = self.sequence_list[seq_name]
        seq_path = os.path.join(self.root, seq_name)
        length = len([p for p in os.listdir(os.path.join(seq_path, 'img'))])
        # print(seq_name,seq_path,length,frame_ids)
        frame_list = [self._get_frame(seq_path, f%length) for f in frame_ids]
        ms_label = [self._get_ms_label(seq_path, f%length) for f in frame_ids]
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
        temp = [0 if i==1 else i for i in ms_label]
        ms_label = [1 if i==2 else i for i in temp]
        return frame_list, anno_frames, object_meta, torch.from_numpy(np.array(ms_label)).float()
