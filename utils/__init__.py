# -*- coding: utf-8 -*-
import logging
import os

import yaml


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False, dim=0, proj=False, is_distribution=False):
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks' Name is wrong, check net config, "
                             f"expected: {model_name_list} received: {net_name}")
        return models.__dict__[net_name]

    if net_name == 'WideResNet':
        import models.nets.wrn as net
        builder = getattr(net, 'build_WideResNet')()
    elif net_name == 'WideResNetVar':
        import models.nets.wrn_var as net
        builder = getattr(net, 'build_WideResNetVar')()
    elif net_name == 'ResNet50':
        import models.nets.resnet50 as net
        builder = getattr(net, 'build_ResNet50')(is_remix, dim, proj, is_distribution)
    else:
        assert Exception("Not Implemented Error")

    if net_name != 'ResNet50':
        setattr_cls_from_kwargs(builder, net_conf)
    return builder.build


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
