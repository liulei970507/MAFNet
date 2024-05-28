# coding=UTF-8
from . import BaseActor
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from random import choice
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    

    #给每个核函数乘上系数
    # B=[]
    # B=[0.02,0.03,0.09,0.12,0.14,0.15,0.15,0.14,0.10,0.05,0.01]
    # kernel_val=[ kernel_value*B[i] for i,kernel_value in enumerate(kernel_val) ]

    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

class DiMPActor_MMD(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred, train_rgb2, train_nir2, train_mid2, train_rgb3, train_nir3, train_mid3, test_rgb2, test_nir2, test_mid2, test_rgb3, test_nir3, test_mid3= \
                                            self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # import pdb
        # pdb.set_trace()
        loss_train_mid2_rgb = mmd_rbf(train_mid2.view(train_mid2.shape[0],-1), train_rgb2.view(train_rgb2.shape[0],-1))# + mmd_rbf(train_mid2, train_nir2)
        loss_train_mid3_rgb = mmd_rbf(train_mid3.view(train_mid3.shape[0],-1), train_rgb3.view(train_rgb3.shape[0],-1))# + mmd_rbf(train_mid3, train_nir3)
        loss_test_mid2_rgb = mmd_rbf(test_mid2.view(test_mid2.shape[0],-1), test_rgb2.view(test_rgb2.shape[0],-1))# + mmd_rbf(test_mid2, test_nir2)
        loss_test_mid3_rgb = mmd_rbf(test_mid3.view(test_mid3.shape[0],-1), test_rgb3.view(test_rgb3.shape[0],-1))# + mmd_rbf(test_mid3, test_nir3)

        loss_train_mid2_nir = mmd_rbf(train_mid2.view(train_mid2.shape[0],-1), train_nir2.view(train_nir2.shape[0],-1))
        loss_train_mid3_nir = mmd_rbf(train_mid3.view(train_mid3.shape[0],-1), train_nir3.view(train_nir3.shape[0],-1))
        loss_test_mid2_nir = mmd_rbf(test_mid2.view(test_mid2.shape[0],-1), test_nir2.view(test_nir2.shape[0],-1))
        loss_test_mid3_nir = mmd_rbf(test_mid3.view(test_mid3.shape[0],-1), test_nir3.view(test_nir3.shape[0],-1))


        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + \
        loss_train_mid2_rgb + loss_train_mid3_rgb + loss_test_mid2_rgb + loss_test_mid3_rgb + \
        loss_train_mid2_nir + loss_train_mid3_nir + loss_test_mid2_nir + loss_test_mid3_nir

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        
        stats['loss_train_mid2_rgb'] = loss_train_mid2_rgb.item()
        stats['loss_train_mid2_nir'] = loss_train_mid2_nir.item()
        stats['loss_train_mid3_rgb'] = loss_train_mid3_rgb.item()
        stats['loss_train_mid3_nir'] = loss_train_mid3_nir.item()
        stats['loss_test_mid2_rgb'] = loss_test_mid2_rgb.item()
        stats['loss_test_mid2_nir'] = loss_test_mid2_nir.item()
        stats['loss_test_mid3_rgb'] = loss_test_mid3_rgb.item()
        stats['loss_test_mid3_nir'] = loss_test_mid3_nir.item()

        return loss, stats


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])
        
        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        return loss, stats

class DiMPActor_random(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight
        self.mp_loss = BCEWithLogitsLoss()
        
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred, mp_train, mp_test = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])
        
        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])
        # 计算模态感知loss
        # (pred_dict["pred_logits"].view(-1), labels)
        # 真值标签
        # mp_loss_train = self.mp_loss(mp_train.squeeze(-1),  data['ms_label_train'].view(-1))
        # mp_loss_test = self.mp_loss(mp_test.squeeze(-1),  data['ms_label_test'].view(-1))
        # 随机标签
        random_label_train = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_train'].view(-1)))])
        random_label_test = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_test'].view(-1)))])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        random_label_train = random_label_train.to(device)
        random_label_test = random_label_train.to(device)
        mp_loss_train = self.mp_loss(mp_train.squeeze(-1), random_label_train.view(-1))
        mp_loss_test = self.mp_loss(mp_test.squeeze(-1), random_label_test.view(-1))
        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + mp_loss_train + mp_loss_test

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        
        stats['mp_loss_train'] = mp_loss_train.item()
        stats['mp_loss_test'] = mp_loss_test.item()
        return loss, stats

class DiMPActor_grl(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight
        self.mp_loss = BCEWithLogitsLoss()
        
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred, mp_train, mp_test = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])
        
        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])
        # 计算模态感知loss
        # (pred_dict["pred_logits"].view(-1), labels)
        # 真值标签
        mp_loss_train = self.mp_loss(mp_train.squeeze(-1).view(-1),  data['ms_label_train'].view(-1))
        mp_loss_test = self.mp_loss(mp_test.squeeze(-1).view(-1),  data['ms_label_test'].view(-1))
        # 随机标签
        # random_label_train = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_train'].view(-1)))])
        # random_label_test = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_test'].view(-1)))])
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # random_label_train = random_label_train.to(device)
        # random_label_test = random_label_train.to(device)
        # mp_loss_train = self.mp_loss(mp_train.squeeze(-1), random_label_train.view(-1))
        # mp_loss_test = self.mp_loss(mp_test.squeeze(-1), random_label_test.view(-1))
        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + self.loss_weight['ms_weight']*(mp_loss_train + mp_loss_test)

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        
        stats['mp_loss_train'] =self.loss_weight['ms_weight']*mp_loss_train.item()
        stats['mp_loss_test'] =self.loss_weight['ms_weight']*mp_loss_test.item()
        return loss, stats

class DiMPActor_grl_sigmoid(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight
        self.mp_loss = BCELoss()
        
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred, mp_train, mp_test = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])
        
        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])
        # 计算模态感知loss
        # (pred_dict["pred_logits"].view(-1), labels)
        # 真值标签
        # import pdb
        # pdb.set_trace()
        mp_loss_train = self.mp_loss(mp_train.squeeze(-1),  data['ms_label_train'].view(-1))
        mp_loss_test = self.mp_loss(mp_test.squeeze(-1),  data['ms_label_test'].view(-1))
        # 随机标签
        # random_label_train = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_train'].view(-1)))])
        # random_label_test = torch.Tensor([choice([0, 1]) for i in range(len(data['ms_label_test'].view(-1)))])
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # random_label_train = random_label_train.to(device)
        # random_label_test = random_label_train.to(device)
        # mp_loss_train = self.mp_loss(mp_train.squeeze(-1), random_label_train.view(-1))
        # mp_loss_test = self.mp_loss(mp_test.squeeze(-1), random_label_test.view(-1))
        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + mp_loss_train + mp_loss_test

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        
        stats['mp_loss_train'] = mp_loss_train.item()
        stats['mp_loss_test'] = mp_loss_test.item()
        return loss, stats
