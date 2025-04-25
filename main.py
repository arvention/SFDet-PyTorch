import os
import torch
import zipfile
import argparse
import numpy as np
import os.path as osp
from solver import Solver
from datetime import datetime
from torch.backends import cudnn
from data.data_loader import get_loader
from utils.genutils import write_print, mkdir


def zip_directory(path, zip_file):
    """Stores all py and cfg project files inside a zip file

    [description]

    Arguments:
        path {string} -- current path
        zip_file {zipfile.ZipFile} -- zip file to contain the project files
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py') or file.endswith('cfg'):
            zip_file.write(osp.join(path, file))
            if file.endswith('cfg'):
                os.remove(file)


def save_config(path,
                version,
                config):
    """saves the configuration of the experiment

    Arguments:
        path {str} -- save path
        version {str} -- version of the model based on the time
        config {dict} -- contains argument and its value

    """
    cfg_name = '{}.{}'.format(version, 'cfg')

    with open(cfg_name, 'w') as f:
        for k, v in config.items():
            f.write('{}: {}\n'.format(str(k), str(v)))

    zip_name = '{}.{}'.format(version, 'zip')
    zip_name = os.path.join(path, zip_name)
    zip_file = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zip_directory('.', zip_file)
    zip_file.close()

    return version


def string_to_boolean(v):
    """Converts string to boolean

    [description]

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        boolean -- boolean true or false
    """
    return v.lower() in ('true')


def main(version, config, output_txt):
    # for fast training
    cudnn.benchmark = True

    data_loader = get_loader(config)
    solver = Solver(version=version,
                    data_loader=data_loader,
                    config=vars(config),
                    output_txt=output_txt)

    if config.mode == 'train':
        temp_save_path = osp.join(config.model_save_path, version)
        mkdir(temp_save_path)
        solver.train()

    elif config.mode == 'test':

        if config.dataset == 'voc':
            temp_save_path = osp.join(config.model_test_path,
                                      config.pretrained_model)
            mkdir(temp_save_path)

        solver.test()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--class_count', type=int, default=21,
                        help='Number of classes in dataset')
    parser.add_argument('--dataset', type=str, default='voc',
                        choices=['voc', 'coco', 'bdd', 'cityscapes'],
                        help='Dataset to use')
    parser.add_argument('--new_size', type=int, default=320,
                        help='New height and width of input images')
    parser.add_argument('--means', type=tuple, default=(104, 117, 123),
                        help='Mean values of the dataset')
    parser.add_argument('--anchor_config', type=str, default='SFDet-300',
                        choices=['SSD-300', 'SSD-512',
                                 'RSSD-300', 'RSSD-512'
                                 'STDN-300',
                                 'SFDet-300', 'SFDet-512'],
                        help='Anchor box configuration to use')
    parser.add_argument('--scale_initial', type=float, default=0.1,  # .07 COCO
                        help='Initial scale of anchor boxes')
    parser.add_argument('--scale_min', type=int, default=0.2,  # .15 COCO
                        help='Minimum scale of anchor boxes in generation')
    parser.add_argument('--scale_max', type=int, default=1.05,  # 1.05 COCO
                        help='Maximum scale of anchor boxes in generation')

    # training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=220,
                        help='Number of epochs')
    # 145, 182, 218 -> 160, 190, 220
    parser.add_argument('--learning_sched', type=list, default=[160, 190],
                        help='List of epochs to reduce the learning rate')
    parser.add_argument('--warmup_epoch', type=int, default=0,
                        help='Number of epochs for warmup')
    parser.add_argument('--sched_gamma', type=float, default=0.1,
                        help='Adjustment gamma for each learning sched')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='Batch size multiplier')

    # architecture settings
    parser.add_argument('--model', type=str, default='SFDet-ResNet',
                        choices=['SFDet-VGG',
                                 'SFDet-ResNet', 'SFDetV2-ResNet'
                                 'SFDet-DenseNet', 'SFDet-ResNeXt',
                                 'SFDet-EfficientNetV2', 'SFDet-MobileNetV3',
                                 'SSD', 'STDN', 'STDN2'],
                        help='Model to use')
    parser.add_argument('--basenet', type=str, default='vgg16_reducedfc.pth',
                        help='Base network for VGG')
    parser.add_argument('--resnet_model', type=str, default='18',
                        choices=['18', '34', '50', '101', '152'],
                        help='ResNet base network configuration')
    parser.add_argument('--densenet_model', type=str, default='121',
                        choices=['121', '169', '201'],
                        help='DenseNet base network configuration')
    parser.add_argument('--resnext_model', type=str, default='50_32x4d',
                        choices=['50_32x4d', '101_32x8d'],
                        help='ResNeXt base network configuration')
    parser.add_argument('--efficientnet_v2_model', type=str, default='s',
                        choices=['s', 'm', 'l'],
                        help='EfficientNet V2 base network configuration')
    parser.add_argument('--mobilenet_v3_model', type=str, default='s',
                        choices=['s', 'l'],
                        help='MobileNet V3 base network configuration')
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='Pre-trained model')
    parser.add_argument('--coco_weights', type=str,
                        default=None,
                        help='COCO trained weights')

    # loss settings
    parser.add_argument('--loss_config', type=str, default='multibox',
                        choices=['multibox'],
                        help='Type of loss')
    parser.add_argument('--pos_neg_ratio', type=int, default=3,
                        help='Ratio for hard negative mining')

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode of execution')
    parser.add_argument('--use_gpu', type=string_to_boolean, default=True,
                        help='Toggles the use of GPU')

    # testing settings
    parser.add_argument('--max_per_image', type=int, default=50,
                        help='Maximum number of detection per image')
    parser.add_argument('--score_threshold', type=float, default=0.01,
                        help='Score threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='NMS threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IOU threshold for detections')

    # pascal voc dataset
    parser.add_argument('--voc_config', type=str, default='0712',
                        choices=['0712', '0712+'],
                        help='Pascal VOC dataset configuration')
    parser.add_argument('--voc_data_path', type=str,
                        default='../../Datasets/PascalVOC/',
                        help='Pascal VOC dataset path')
    parser.add_argument('--use_07_metric', type=string_to_boolean,
                        default=True,
                        help='Toggles the VOC2007 11-point metric')

    # coco dataset
    parser.add_argument('--coco_year', type=str, default='2017',
                        choices=['2017'],
                        help='COCO dataset configuration')
    parser.add_argument('--coco_data_path', type=str,
                        default='../../Datasets/Coco/',
                        help='COCO dataset path')

    # bdd dataset
    parser.add_argument('--bdd_version', type=str, default='100k',
                        choices=['100k'],
                        help='BDD dataset version')
    parser.add_argument('--bdd_data_path', type=str,
                        default='../../Datasets/bdd100k/',
                        help='BDD dataset path')

    # cityscapes dataset
    parser.add_argument('--cityscapes_data_path', type=str,
                        default='../../Datasets/Cityscapes/',
                        help='Cityscapes dataset path')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')
    parser.add_argument('--model_test_path', type=str, default='./tests',
                        help='Path for saving results')
    parser.add_argument('--model_eval_path', type=str, default='./eval',
                        help='Path for saving results')

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=1,
                        help='Number of steps for logging loss')
    parser.add_argument('--model_save_step', type=int, default=5,
                        help='Number of step for saving model')

    config = parser.parse_args()

    args = vars(config)
    output_txt = ''

    if args['mode'] == 'train':
        version = str(datetime.now()).replace(':', '_')
        version = '{}_train'.format(version)
        path = args['model_save_path']
        path = osp.join(path, version)
        output_txt = osp.join(path, '{}.txt'.format(version))

    elif args['mode'] == 'test':
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = osp.join(path, model[0])
        output_txt = osp.join(path, '{}.txt'.format(version))

    mkdir(path)
    save_config(path, version, args)

    write_print(output_txt, '------------ Options -------------')
    for k, v in args.items():
        write_print(output_txt, '{}: {}'.format(str(k), str(v)))
    write_print(output_txt, '-------------- End ----------------')

    main(version, config, output_txt)
