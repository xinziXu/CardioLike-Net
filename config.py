# -*- coding: utf-8 -*-
"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""

import argparse
from pickle import TRUE

from numpy import True_

arg_lists = []
parser = argparse.ArgumentParser(description='unet_segmentation')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--num_classes', type=int, default=4,
                      help='Number of classes to classify')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=16,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--pin_memory', type=str2bool, default=True,
                      help='whether to copy tensors into CUDA pinned memory')                      
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train indices')
data_arg.add_argument( "--one_model", "-V", action="store_true", default=False)
data_arg.add_argument( "--only_ii", "-T", action="store_true", default=False)
data_arg.add_argument( "--inter_patient", "-S", action="store_true", default=False)
data_arg.add_argument( "--save_train_test_data", "-x", action="store_true", default=False)
data_arg.add_argument( "--trans_en", "-q", action="store_true", default=False)
data_arg.add_argument( "--weight_en",  action="store_true", default=False)
data_arg.add_argument( "--crf_en", action="store_true", default=False)
data_arg.add_argument( "--fea", action="store_true", default=False)

# data_arg.add_argument( "--only_infer", action="store_true", default=False)

data_arg.add_argument("--db_name",type = str, default='mitdb')
# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum value')
train_arg.add_argument('--epochs', type=int, default=60,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.001,
                       help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=5e-4,
                       help='value of weight dacay for regularization')
train_arg.add_argument('--nesterov', type=str2bool, default=True,
                       help='Whether to use Nesterov momentum')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=100,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--gamma', type=float, default=0.1,
                       help='value of learning rate decay')
train_arg.add_argument('--num_icen', type = int, default=0 )

# train_arg.add_argument('--step', nargs='+', type=int, 
#                        help='value of learning rate decay')
# train_arg.add_argument('--loss_weight', type=float ,nargs='+',
#                        help='value of learning rate decay')                       

train_arg.add_argument('--task',type = str, default='arrhythmia')
train_arg.add_argument('--leads_id', nargs= '+', type=int, default=[0, 1])
train_arg.add_argument('--qat', type = str2bool, default=False)
train_arg.add_argument('--qat_lstm', type = str2bool, default=False)
train_arg.add_argument('--ptsq', type = str2bool, default=False)
train_arg.add_argument('--accelerate', type = str2bool, default=False)
train_arg.add_argument('--ma_lstm', type = str2bool, default=False)
train_arg.add_argument('--ma_decoder', type = str2bool, default=False)
# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument(
      "--device", type=str, default='cuda:0',
      help="Number of filters per layer.")                      
misc_arg.add_argument('--best', type=str2bool, default=False,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/cifar100',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')
misc_arg.add_argument('--model_name', type=str, default='bilstm_atten',
                      help='Name of the model to save as')
misc_arg.add_argument('--loss_type', type=str, default='bce',
                      help='Name of the lossl to save as')
misc_arg.add_argument('--model_num', type=int, default=1,
                      help='Number of models to train for DML')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
