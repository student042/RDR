# import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, EMA, get_optimizer, get_cosine_schedule_with_warmup
from models.main.main import S2_VER
from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader, Emotion_SSL_Dataset
from datasets.data_utils import get_data_loader
from data_split_utils import prepare_stratified_train_val_split

'''
FI数据集与cifar10的切换: net部分，调整stride；数据集部分，SSL/EmotionSSL
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    args.overwrite = True
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


# def main_worker(gpu, ngpus_per_node, args):
def main_worker(gpu, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    logger.info(f"  Task = {args.dataset}@{args.num_labels}")

    if args.val_ratio > 0:
        source_train_json = (
            args.train_json_path
            if args.train_json_path
            else os.path.join(args.train_data_dir, "train.json")
        )
        split_dir = args.split_dir
        if not split_dir:
            ratio_tag = int(round(args.val_ratio * 100))
            split_dir = os.path.join(
                "splits",
                f"{args.dataset}_seed{args.val_seed}_val{ratio_tag:02d}",
            )
        train_json_path, val_json_path, split_manifest = (
            prepare_stratified_train_val_split(
                source_train_json=source_train_json,
                output_dir=split_dir,
                val_ratio=args.val_ratio,
                seed=args.val_seed,
                stratified=args.stratified_val_split,
            )
        )
        args.train_json_path = train_json_path
        args.val_json_path = val_json_path
        args.val_data_dir = args.train_data_dir
        logger.info(
            "[DATA-SPLIT] source=%s train=%s val=%s counts=%s/%s "
            "class_counts(train)=%s class_counts(val)=%s",
            source_train_json,
            train_json_path,
            val_json_path,
            split_manifest["train_count"],
            split_manifest["val_count"],
            split_manifest["train_class_counts"],
            split_manifest["val_class_counts"],
        )
    elif args.val_json_path and not args.val_data_dir:
        args.val_data_dir = args.train_data_dir

    # SET CoMatch: class CoMatch in models.comatch
    args.bn_momentum = 1.0 - 0.999
    
    _net_builder = net_builder('ResNet50', False, None, is_remix=False, dim=args.low_dim, proj=True)

    model = S2_VER(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     tb_log=tb_log,
                     args=args,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter*args.epoch,
                                                num_warmup_steps=args.num_train_iter * 0)
    ## set SGD and cosine lr on CoMatch 
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.model = model.model.cuda(args.gpu)

    else:
        model.model = torch.nn.DataParallel(model.model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Dataset & DataLoader
    if args.dataset != "imagenet":

        train_dset = Emotion_SSL_Dataset(args, alg='comatch', name=args.dataset, train=True,
                    num_classes=args.num_classes, data_dir=args.train_data_dir, split='train')
        lb_dset, ulb_dset = train_dset.get_ssl_dset(
            args.num_labels,
            include_lb_to_ulb=getattr(args, 'include_lb_to_ulb', True),
        )

        eval_data_dir = args.val_data_dir if args.val_data_dir else args.test_data_dir
        eval_split = 'val' if args.val_data_dir else 'test'
        _eval_dset = Emotion_SSL_Dataset(args, alg='comatch', name=args.dataset, train=False,
                    num_classes=args.num_classes, data_dir=eval_data_dir, split=eval_split)

        eval_dset = _eval_dset.get_dset()
        test_dset = None
    else:
        image_loader = ImageNetLoader(root_path=args.data_dir, num_labels=args.num_labels,
                                      num_class=args.num_classes)
        lb_dset = image_loader.get_lb_train_data()
        ulb_dset = image_loader.get_ulb_train_data()
        eval_dset = image_loader.get_lb_test_data()
        test_dset = None
    
    print(len(lb_dset), len(ulb_dset), len(eval_dset))
                            
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler=args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size * args.uratio,
                                               data_sampler=args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=args.num_workers)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=False)
    print(len(loader_dict['train_lb']), len(loader_dict['train_ulb']), len(loader_dict['eval']))

    ## set DataLoader on CoMatch
    model.set_data_loader(loader_dict)
    model.set_dset(ulb_dset)
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    if getattr(args, 'role_diag_only', False):
        if not args.resume or not args.load_path:
            raise ValueError("--role_diag_only requires --resume and --load_path")
        role_diag_loader = get_data_loader(
            ulb_dset,
            args.eval_batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        output_path = args.role_diag_output
        if not output_path:
            output_path = os.path.join(save_path, "role_reliability_diag.jsonl")
        model.dump_role_reliability_diag(
            diag_loader=role_diag_loader,
            args=args,
            output_path=output_path,
        )
        logger.info("[ROLE-DIAG] output=%s", output_path)
        return

    # START TRAINING of CoMatch
    trainer = model.train
    best_eval_acc = 0
    for epoch in range(args.epoch):
        eval_acc = trainer(args, epoch, best_eval_acc, logger=logger)
        best_eval_acc = max(eval_acc, best_eval_acc)

    if args.eval_on_test_final_only:
        if not args.val_data_dir:
            raise ValueError(
                "--eval_on_test_final_only requires a validation split"
            )
        best_path = os.path.join(save_path, 'model_best.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError(
                f"Best validation checkpoint was not created: {best_path}"
            )
        final_test_dset = Emotion_SSL_Dataset(
            args,
            alg='comatch',
            name=args.dataset,
            train=False,
            num_classes=args.num_classes,
            data_dir=args.test_data_dir,
            split='test',
        ).get_dset()
        final_test_loader = get_data_loader(
            final_test_dset,
            args.eval_batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        model.load_model(best_path)
        model.ema = EMA(model.model, model.ema_m)
        model.ema.register()
        model.ema.load(model.ema_model)
        final_test = model.evaluate(
            eval_loader=final_test_loader,
            args=args,
            epoch=None,
            split_name='test',
        )
        if args.export_best_epoch_analysis:
            model._save_evaluation_analysis(
                eval_dict=final_test,
                save_path=save_path,
                file_prefix="best_checkpoint_test",
                epoch=model.best_epoch,
                split_name="test",
            )
        logger.info(
            "[EVAL-TEST-FINAL] checkpoint=%s loss=%.6f acc=%.6f "
            "macro_f1=%.6f positive_f1=%.6f negative_f1=%.6f "
            "neutral_f1=%.6f weighted_f1=%.6f class_precision=%s "
            "class_recall=%s class_f1=%s confusion_matrix=%s",
            best_path,
            final_test['eval/loss'],
            final_test['eval/top-1-acc'],
            final_test['eval/macro-f1'],
            final_test['eval/class-f1'][0],
            final_test['eval/class-f1'][1],
            final_test['eval/class-f1'][2],
            final_test['eval/weighted-f1'],
            final_test['eval/class-precision'],
            final_test['eval/class-recall'],
            final_test['eval/class-f1'],
            final_test['eval/confusion-matrix'],
        )

    # logging.warning(f"GPU {args.rank} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='eccv_result/single/sgd/add')
    parser.add_argument('-sn', '--save_name', type=str, default='main')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--role_diag_only', type=str2bool, default=False,
                        help='load a checkpoint and dump HH/HL/LH/LL role-reliability diagnostics on train_ulb')
    parser.add_argument('--role_diag_output', type=str, default='',
                        help='jsonl path for role-dependent reliability diagnostics')

    '''
    Training Configuration of main
    '''

    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--num_train_iter', type=int, default=1024,
                        help='total number of training iterations')
    parser.add_argument('-nl', '--num_labels', type=int, default=3)
    parser.add_argument('-bsz', '--batch_size', type=int, default=2)
    parser.add_argument('--uratio', type=int, default=4,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    ''' Comatch parameters '''

    parser.add_argument('--hard_label', type=str2bool, default=True)
    # comatch 默认温度是0.2
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--p_cutoff', type=float, default=0.95, help='pseudo label threshold')
    parser.add_argument('--noise_th', default=0.3, type=float, help='graph noise threshold')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--ldl_ratio', type=float, default=0.2)
    parser.add_argument('--low_dim', type=int, default=2816)
    parser.add_argument('--lam_c', type=float, default=3, help='coefficient of contrastive loss')
    parser.add_argument('--lam_d', type=float, default=3, help='coefficient of distribution loss')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--dynamic_th', type=float, default=0.7)
    parser.add_argument('--dis_ce', action='store_true')
    parser.add_argument('--update_m', type=str, default='L2')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--add_ulb', type=str2bool, default=False)
    # parser.add_argument('--class_weight', type=list, default=[0.2, 0.5, 0.3])

    '''
    MLLM-assisted pseudo-label reliability verification.
    MLLM evidence is only used as a hard verifier for SCRD pseudo-labels;
    it never replaces SCRD pseudo-label targets.
    '''
    parser.add_argument('--use_mllm_verification', type=str2bool, default=False)
    parser.add_argument('--mllm_evidence_path', type=str, default='')
    parser.add_argument('--mllm_verify_mode', type=str, default='mm',
                        choices=['off', 'mm', 'tri', 'strict', 'img_text'])
    parser.add_argument('--mllm_label_map', type=str,
                        default='positive:0,negative:1,neutral:2')
    parser.add_argument('--mllm_missing_policy', type=str, default='reject',
                        choices=['reject', 'pass'])
    parser.add_argument('--mllm_evidence_shuffle', type=str2bool, default=False)
    parser.add_argument('--mllm_gate_sim_loss', type=str2bool, default=True)
    parser.add_argument('--mllm_selective_verify', type=str2bool, default=True)
    parser.add_argument('--mllm_verify_policy', type=str, default='risky_only',
                        choices=['risky_only', 'all'])
    parser.add_argument('--mllm_action', type=str, default='veto',
                        choices=[
                            'veto',
                            'soft_weight',
                            'override',
                            'ecs_v1',
                            'hybrid_cerw',
                            'dctr_plf',
                            'risk_veto',
                            'risk_soft_weight',
                        ])
    parser.add_argument('--mllm_soft_weight', type=float, default=0.5)
    parser.add_argument('--mllm_ecs_conf_threshold', type=float, default=0.75)
    parser.add_argument('--hybrid_support_high', type=float, default=0.65)
    parser.add_argument('--hybrid_support_low', type=float, default=0.30)
    parser.add_argument('--hybrid_qwen_conf_high', type=float, default=0.75)
    parser.add_argument('--hybrid_weight_agree', type=float, default=1.0)
    parser.add_argument('--hybrid_weight_uncertain', type=float, default=0.5)
    parser.add_argument('--hybrid_weight_high_conflict', type=float, default=0.0)
    parser.add_argument('--hybrid_weight_opposite_conflict', type=float, default=0.0)
    parser.add_argument('--hybrid_neutral_max_weight', type=float, default=0.5)
    parser.add_argument('--hybrid_use_distribution', type=str2bool, default=True)
    parser.add_argument('--hybrid_use_opposite_conflict', type=str2bool, default=True)
    parser.add_argument('--dctr_prior_mean', type=float, default=0.5,
                        help='Beta prior mean for DCTR internal/external calibration')
    parser.add_argument('--dctr_prior_strength', type=float, default=6.0,
                        help='Beta prior strength for low-label DCTR calibration')
    parser.add_argument('--dctr_min_reliability', type=float, default=0.05,
                        help='minimum continuous branch reliability kept by DCTR-PLF')
    parser.add_argument('--dctr_apply_all', type=str2bool, default=True,
                        help='apply DCTR reliability to all SCRD candidates instead of only risky samples')
    parser.add_argument('--dctr_use_aug_stability', type=str2bool, default=True)
    parser.add_argument('--dctr_use_js_similarity', type=str2bool, default=True)
    parser.add_argument('--use_dctr_msg', type=str2bool, default=False,
                        help='enable DCTR reliability-guided soft modality selection')
    parser.add_argument('--lambda_dctr_msg', type=float, default=0.03)
    parser.add_argument('--dctr_msg_warmup_epoch', type=int, default=20)
    parser.add_argument('--dctr_msg_rampup_epoch', type=int, default=20)
    parser.add_argument('--dctr_msg_reliability_threshold', type=float, default=0.45)
    parser.add_argument('--dctr_msg_prior_temperature', type=float, default=0.5)
    parser.add_argument('--dctr_msg_labeled_temperature', type=float, default=0.5)
    parser.add_argument('--dctr_msg_smoothing', type=float, default=0.05)
    parser.add_argument('--dctr_msg_min_margin', type=float, default=0.02)
    parser.add_argument('--dctr_msg_min_samples', type=int, default=2)
    parser.add_argument('--dctr_msg_labeled_weight', type=float, default=1.0)
    parser.add_argument('--dctr_msg_unlabeled_weight', type=float, default=1.0)
    parser.add_argument('--dctr_msg_use_labeled_anchor', type=str2bool, default=True)
    parser.add_argument('--dctr_msg_exclude_opposite_conflict', type=str2bool,
                        default=False)
    parser.add_argument('--dctr_msg_reliability_shuffle', type=str2bool,
                        default=False,
                        help='shuffle only DCTR-MSG reliability targets')
    parser.add_argument('--use_soft_modal_selector', type=str2bool, default=False,
                        help='replace hard argmax modality selection with differentiable soft fusion')
    parser.add_argument('--modal_selector_temperature', type=float, default=1.0)
    parser.add_argument('--use_ucrf', type=str2bool, default=False,
                        help='enable utility-calibrated residual fusion')
    parser.add_argument('--lambda_ucrf', type=float, default=0.05)
    parser.add_argument('--ucrf_warmup_epoch', type=int, default=20,
                        help='labeled-only selector warmup before residual and unlabeled UCRF')
    parser.add_argument('--ucrf_rampup_epoch', type=int, default=20)
    parser.add_argument('--ucrf_residual_beta', type=float, default=0.30,
                        help='maximum residual correction strength')
    parser.add_argument('--ucrf_utility_temperature', type=float, default=0.50)
    parser.add_argument('--ucrf_reliability_threshold', type=float, default=0.75)
    parser.add_argument('--ucrf_min_samples', type=int, default=2)
    parser.add_argument('--ucrf_labeled_weight', type=float, default=1.0)
    parser.add_argument('--ucrf_unlabeled_weight', type=float, default=0.5)
    parser.add_argument('--ucrf_use_unlabeled', type=str2bool, default=True)
    parser.add_argument('--use_msd', type=str2bool, default=False,
                        help='enable evidence-calibrated modality support distillation')
    parser.add_argument('--lambda_msd', type=float, default=0.1)
    parser.add_argument('--msd_warmup_epoch', type=int, default=20)
    parser.add_argument('--msd_rampup_epoch', type=int, default=20)
    parser.add_argument('--msd_reliability_threshold', type=float, default=0.75)
    parser.add_argument('--msd_support_temperature', type=float, default=0.5)
    parser.add_argument('--msd_labeled_temperature', type=float, default=0.5)
    parser.add_argument('--msd_support_smoothing', type=float, default=0.05)
    parser.add_argument('--msd_labeled_weight', type=float, default=1.0)
    parser.add_argument('--msd_unlabeled_weight', type=float, default=1.0)
    parser.add_argument('--msd_use_labeled_anchor', type=str2bool, default=True)
    parser.add_argument('--msd_use_unlabeled_evidence', type=str2bool, default=True)
    parser.add_argument('--msd_exclude_opposite_conflict', type=str2bool, default=True)
    parser.add_argument('--msd_exclude_neutral_consensus', type=str2bool, default=False)
    parser.add_argument('--msd_prior_mode', type=str, default='evidence',
                        choices=['evidence', 'uniform', 'random'])
    parser.add_argument('--msd_evidence_shuffle', type=str2bool, default=False,
                        help='shuffle only the MSD teacher evidence within each batch')
    parser.add_argument('--use_ce_umc', type=str2bool, default=False,
                        help='enable calibrated external unimodal soft supervision')
    parser.add_argument('--lambda_ce_umc_text', type=float, default=0.02)
    parser.add_argument('--lambda_ce_umc_image', type=float, default=0.05)
    parser.add_argument('--ce_umc_warmup_epoch', type=int, default=10)
    parser.add_argument('--ce_umc_rampup_epoch', type=int, default=20)
    parser.add_argument('--ce_umc_target_temperature', type=float, default=2.0)
    parser.add_argument('--ce_umc_text_conf_threshold', type=float, default=0.90)
    parser.add_argument('--ce_umc_image_conf_threshold', type=float, default=0.90)
    parser.add_argument('--ce_umc_min_certainty', type=float, default=0.20)
    parser.add_argument('--ce_umc_confidence_power', type=float, default=1.0)
    parser.add_argument('--ce_umc_min_samples', type=int, default=2)
    parser.add_argument('--ce_umc_scope', type=str, default='all',
                        choices=['all', 'plf_accepted', 'plf_rejected'],
                        help='which unlabeled samples receive CE-UMC supervision')
    parser.add_argument('--ce_umc_evidence_shuffle', type=str2bool, default=False,
                        help='shuffle CE-UMC evidence only, leaving EC-PLF evidence unchanged')
    parser.add_argument('--use_ec_pfd', type=str2bool, default=False,
                        help='enable evidence-calibrated pseudo-label feedback disentanglement')
    parser.add_argument('--lambda_ec_pfd', type=float, default=0.2)
    parser.add_argument('--ec_pfd_threshold', type=float, default=0.75)
    parser.add_argument('--ec_pfd_use_soft_label', type=str2bool, default=True)
    parser.add_argument('--ec_pfd_detach_target', type=str2bool, default=True)
    parser.add_argument('--ec_pfd_min_samples', type=int, default=1)
    parser.add_argument('--use_ead_a', type=str2bool, default=False,
                        help='enable evidence-adaptive disentanglement alignment')
    parser.add_argument('--lambda_ead_a', type=float, default=0.02)
    parser.add_argument('--ead_a_warmup_epoch', type=int, default=30)
    parser.add_argument('--ead_a_reliability_threshold', type=float, default=0.45)
    parser.add_argument('--ead_a_min_samples', type=int, default=2)
    parser.add_argument('--ead_a_common_align_weight', type=float, default=0.2)
    parser.add_argument('--ead_a_conflict_boost', type=float, default=0.5)
    parser.add_argument('--ead_a_neutral_scale', type=float, default=0.5)
    parser.add_argument('--ead_a_opposite_common_scale', type=float, default=0.0)
    parser.add_argument('--ead_a_evidence_shuffle', type=str2bool, default=False,
                        help='shuffle only EAD-A evidence within each batch')
    parser.add_argument('--use_sa_dd', type=str2bool, default=False,
                        help='enable stability-aware relation-adaptive dynamic disentanglement')
    parser.add_argument('--sa_dd_version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='v2 adds local residual regularization; v1 preserves legacy behavior')
    parser.add_argument('--lambda_sa_dd', type=float, default=0.01)
    parser.add_argument('--sa_dd_warmup_epoch', type=int, default=30)
    parser.add_argument('--sa_dd_rampup_epoch', type=int, default=20)
    parser.add_argument('--sa_dd_history_momentum', type=float, default=0.9)
    parser.add_argument('--sa_dd_min_history_stability', type=float, default=0.6)
    parser.add_argument('--sa_dd_min_aug_stability', type=float, default=0.6)
    parser.add_argument('--sa_dd_min_relation_conf', type=float, default=0.2)
    parser.add_argument('--sa_dd_v2_min_gate', type=float, default=0.05,
                        help='minimum local intervention score used by SA-DD-v2')
    parser.add_argument('--sa_dd_v2_stability_temperature', type=float, default=0.1,
                        help='temperature for the SA-DD-v2 soft stability gate')
    parser.add_argument('--sa_dd_relation_floor', type=float, default=0.2)
    parser.add_argument('--sa_dd_relation_ceiling', type=float, default=0.8)
    parser.add_argument('--sa_dd_common_weight', type=float, default=1.0)
    parser.add_argument('--sa_dd_private_weight', type=float, default=1.0)
    parser.add_argument('--sa_dd_ecplf_floor', type=float, default=0.5,
                        help='minimum relation trust for samples rejected by EC-PLF')
    parser.add_argument('--sa_dd_neutral_scale', type=float, default=0.5)
    parser.add_argument('--sa_dd_min_samples', type=int, default=2)
    parser.add_argument('--sa_dd_replace_unlabeled_sim', type=str2bool,
                        default=True,
                        help='legacy v1 only: replace SCRD unlabeled similarity pairs')
    parser.add_argument('--sa_dd_class_momentum', type=float, default=0.95)
    parser.add_argument('--sa_dd_max_class_share', type=float, default=0.85)
    parser.add_argument('--sa_dd_min_class_share', type=float, default=0.02)
    parser.add_argument('--sa_dd_class_guard_min_count', type=int, default=64)
    parser.add_argument('--sa_dd_evidence_shuffle', type=str2bool, default=False,
                        help='shuffle only SA-DD relation evidence within each batch')
    parser.add_argument('--sa_dd_ablation', type=str, default='none',
                        choices=['none', 'wo_relation', 'wo_stability',
                                 'fixed_gate', 'unified_reliability'],
                        help='SA-DD/RAD mechanism ablation without changing EC-PLF')
    parser.add_argument('--sa_dd_fixed_gate_value', type=float, default=0.5,
                        help='constant gate used when --sa_dd_ablation fixed_gate')
    parser.add_argument('--risk_margin_threshold', type=float, default=0.2)
    parser.add_argument('--risk_kl_threshold', type=float, default=0.5)


    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='Resnet50')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--train_data_dir', type=str, default='')
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--test_data_dir', type=str, default='')
    parser.add_argument('--train_json_path', type=str, default='')
    parser.add_argument('--val_json_path', type=str, default='')
    parser.add_argument('--test_json_path', type=str, default='')
    parser.add_argument('--labeled_ids_path', type=str, default='')
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default=True)
    parser.add_argument('--val_ratio', type=float, default=0.0)
    parser.add_argument('--val_seed', type=int, default=42)
    parser.add_argument('--stratified_val_split', type=str2bool, default=True)
    parser.add_argument('--split_dir', type=str, default='')
    parser.add_argument('--eval_on_test_final_only', type=str2bool, default=False)
    parser.add_argument('--log_pseudo_diag', type=str2bool, default=True)
    parser.add_argument('--log_confusion_matrix', type=str2bool, default=True)
    parser.add_argument('--export_best_epoch_analysis', type=str2bool, default=True,
                        help='export best validation epoch metrics and per-sample predictions')
    parser.add_argument('-ds', '--dataset', type=str, default='mvsa-s')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    if args.use_ce_umc:
        if not args.mllm_evidence_path:
            parser.error("--use_ce_umc requires --mllm_evidence_path")
        if args.lambda_ce_umc_text < 0.0 or args.lambda_ce_umc_image < 0.0:
            parser.error("CE-UMC loss weights must be non-negative")
        if args.ce_umc_target_temperature <= 0.0:
            parser.error("--ce_umc_target_temperature must be positive")
        if not 0.0 <= args.ce_umc_text_conf_threshold <= 1.0:
            parser.error("--ce_umc_text_conf_threshold must be in [0, 1]")
        if not 0.0 <= args.ce_umc_image_conf_threshold <= 1.0:
            parser.error("--ce_umc_image_conf_threshold must be in [0, 1]")
        if not 0.0 <= args.ce_umc_min_certainty <= 1.0:
            parser.error("--ce_umc_min_certainty must be in [0, 1]")
        if args.ce_umc_confidence_power < 0.0:
            parser.error("--ce_umc_confidence_power must be non-negative")
        if args.ce_umc_min_samples < 1:
            parser.error("--ce_umc_min_samples must be at least 1")
    if args.use_sa_dd and not args.mllm_evidence_path:
        parser.error("--use_sa_dd requires --mllm_evidence_path")
    if args.use_sa_dd and args.use_ead_a:
        parser.error(
            "--use_sa_dd and --use_ead_a are alternative disentanglement "
            "strategies; enable only one"
        )
    if args.use_dctr_msg:
        if not args.use_mllm_verification or args.mllm_action != "dctr_plf":
            parser.error(
                "--use_dctr_msg requires --use_mllm_verification 1 and "
                "--mllm_action dctr_plf"
            )
        if not args.use_soft_modal_selector:
            parser.error("--use_dctr_msg requires --use_soft_modal_selector 1")
        if args.use_msd:
            parser.error(
                "--use_dctr_msg and --use_msd supervise the same modality "
                "selector; enable only one"
            )
    if args.use_ucrf:
        if not 0.0 <= args.ucrf_residual_beta <= 1.0:
            parser.error("--ucrf_residual_beta must be in [0, 1]")
        if args.lambda_ucrf < 0.0:
            parser.error("--lambda_ucrf must be non-negative")
        if args.ucrf_utility_temperature <= 0.0:
            parser.error("--ucrf_utility_temperature must be positive")
        if args.use_msd or args.use_dctr_msg:
            parser.error(
                "--use_ucrf cannot be combined with --use_msd or "
                "--use_dctr_msg because they supervise the same selector"
            )
        if args.ucrf_use_unlabeled and (
            not args.use_mllm_verification
            or args.mllm_action != "dctr_plf"
        ):
            parser.error(
                "unlabeled UCRF requires --use_mllm_verification 1 and "
                "--mllm_action dctr_plf; otherwise set "
                "--ucrf_use_unlabeled 0"
            )
    main(args)
