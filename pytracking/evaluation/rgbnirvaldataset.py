import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os

class RGBNIRVALDataset(BaseDataset):
    
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbnir_path
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
        return Sequence(sequence_info['name'], frames, 'rgbnirval', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_list(self):
        # sequence_list = ['100', '101', '103', '107', '108', '109', '112', '114', '115', '116', '118', '119', '120', '122', '124', '127', '128', '129', '12', '131', '132', '135', '137', '139', '140', '141', '142', '143', '145', '147', '150', '151', '153', '157', '159', '160', '162', '163', '166', '167', '174', '176', '179', '188', '191', '192', '195', '197', '198', '19', '202', '203', '204', '206', '209', '20', '210', '211', '214', '21', '220', '222', '223', '226', '22', '237', '243', '244', '245', '247', '249', '251', '252', '254', '255', '256', '258', '260', '262', '263', '264', '266', '26', '275', '277', '278', '279', '281', '282', '284', '285', '287', '288', '28', '290', '295', '296', '29', '2', '306', '307', '31', '320', '322', '323', '326', '32', '332', '335', '336', '337', '338', '33', '340', '341', '343', '347', '348', '349', '350', '352', '356', '361', '367', '370', '373', '374', '376', '381', '382', '384', '385', '386', '38', '391', '394', '396', '397', '398', '400', '403', '405', '406', '408', '409', '40', '410', '411', '412', '414', '415', '416', '417', '419', '41', '420', '423', '424', '426', '427', '430', '434', '435', '438', '439', '43', '440', '445', '446', '447', '44', '451', '452', '459', '45', '462', '467', '473', '479', '47', '480', '481', '482', '483', '489', '490', '492', '495', '497', '500', '50', '51', '55', '57', '58', '59', '60', '61', '64', '67', '69', '6', '70', '76', '7', '80', '81', '84', '8', '90', '92', '96', '98', '99']
        # sequence_list = ['85', '133', '136', '182', '186', '189', '190', '267', '315', '517', '523', '526', '614', '628', '663', '664', '665', '666', '667', '668', '669', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999', '1000']
        sequence_list = ['998', '517', '951', '986', '943', '958', '899', '923', '190', '981', '904', '687', '945', '675', '964', '956', '963', '971']
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = '/DATA/RGBNIR_PLUS/'+sequence_info["name"]
            sequence_info["anno_path"] = sequence_info["path"]+'/'+'refine_'+sequence_list[i]+'_groundtruth_rect.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
