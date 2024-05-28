import torch.nn as nn
import torch.optim as optim
from ltr.dataset import RGBNIR, RGBNIRPLUS
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet_ms_morep_bn
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU


def run(settings):
    settings.description = 'Default train settings for DiMP with ResNet50 as backbone.'
    settings.batch_size = 28
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 5
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    # settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/clf_ce', 'ClfTrain/test_loss']

    # Train datasets
    rgbnir_train = RGBNIR(settings.env.rgbnir_dir, split='train')
    rgbnirplus_train = RGBNIRPLUS(settings.env.rgbnirplus_dir, split='train')
    # Validation datasets
    rgbnir_val = RGBNIR(settings.env.rgbnir_dir, split='test')
    rgbnirplus_val = RGBNIRPLUS(settings.env.rgbnirplus_dir, split='test')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    data_processing_train = processing.DiMPProcessing_MS(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.DiMPProcessing_MS(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    label_function_params=label_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.DiMPSampler_MS([rgbnir_train, rgbnirplus_train], [2.5, 1],
                                        samples_per_epoch=26000, max_gap=30, num_test_frames=3, num_train_frames=3,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.DiMPSampler_MS([rgbnir_val, rgbnirplus_val], [1.2, 1], samples_per_epoch=5000, max_gap=30,
                                      num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network and actor
    net = dimpnet_ms_morep_bn.dimpnet50(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512,
                            optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'iou': nn.MSELoss(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    loss_weight = {'iou': 1, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}

    actor = actors.DiMPActor_grl(net=net, objective=objective, loss_weight=loss_weight)
    lrgain = 1e-1
    # Optimizer
    optimizer = optim.Adam([{'params': actor.net.mp_layer.parameters(), 'lr': 2e-4},
                            {'params': actor.net.rgb_branch_layer2.parameters(), 'lr': 2e-4},
                            {'params': actor.net.rgb_branch_layer3.parameters(), 'lr': 2e-4},
                            {'params': actor.net.nir_branch_layer2.parameters(), 'lr': 2e-4},
                            {'params': actor.net.nir_branch_layer3.parameters(), 'lr': 2e-4},
                            {'params': actor.net.classifier.filter_initializer.parameters(), 'lr': lrgain*5e-5},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': lrgain*5e-4},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': lrgain*5e-5},
                            {'params': actor.net.bb_regressor.parameters()},
                            {'params': actor.net.feature_extractor.parameters(), 'lr': lrgain*2e-5}],
                           lr=lrgain*2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(50, load_latest=True, fail_safe=True)
