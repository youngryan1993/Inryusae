import torch
import numpy as np
from .dataset import data_provider
import argparse
import time
import os
from torchvision import transforms
from models import PBML, inceptionv3

# Get Arguments
DATA_SET = 'cub200_2011'
LOSS_TYPE = 'NpairLoss'
LOG_SAVE_PATH = './tensorboard_log/'
FORMER_CKPT = '02-07-14-27/model.ckpt-27900'
CKPT_PATH = './formerTrain/'
# To approximately reduce the mean of input images
image_mean = np.array([123, 117, 104], dtype=np.float32)  # RGB
# To shape the array image_mean to (1, 1, 1, 3) => three channels
image_mean = image_mean[None, None, None, [2, 1, 0]]

neighbours = [1, 2, 4, 8, 16, 32]
products_neighbours = [1, 10, 1000]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def get_arguments():

    parser = argparse.ArgumentParser(description="Get Arguments")
    parser.add_argument("--dataSet", type=str, default=DATA_SET,
                        help="Training on which dataset, cars196, cub200, products")
    parser.add_argument("--LossType", type=str, default=LOSS_TYPE,
                        help="The type of Loss to be used in training")
    parser.add_argument("--log_save_path", type=str, default=LOG_SAVE_PATH,
                        help="Directory to save tenorboardX log files")
    parser.add_argument("--formerTimer", type=str, default=FORMER_CKPT,
                        help="The time that the former checkpoint is created")
    parser.add_argument("--checkpoint_path", type=str, default=CKPT_PATH,
                        help="Directory to restore and save checkpoints")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size, 128 is recommended for cars196")
    parser.add_argument("--Regular_factor", type=float, default=5e-3,
                        help="weight decay factor, we recommend 5e-3 for cars196")
    parser.add_argument("--init_learning_rate", type=float, default=7e-5,
                        help="initial learning rate, we recommend 7e-5 for cars196")
    parser.add_argument("--default_image_size", type=int, default=224,
                        help="The size of input images")
    parser.add_argument("--SaveVal", type=bool, default=True,
                        help="Whether save checkpoint")
    parser.add_argument("--normalize", type=bool, default=True,
                        help="Whether use batch normalization")
    parser.add_argument("--load_formalVal", type=bool, default=False,
                        help="Whether load former value before training")
    parser.add_argument("--embedding_size", type=float, default=128,
                        help="The size of embedding, we recommend 128 for cars196")
    parser.add_argument("--loss_l2_reg", type=float, default=3e-3,
                        help="The factor of embedding l2_loss, we recommend 3e-3 for cars196")
    parser.add_argument("--init_batch_per_epoch", type=int, default=500,
                        help="init_batch_per_epoch, 500 for cars and cub")
    parser.add_argument("--batch_per_epoch", type=int, default=64,
                        help="The number of batches per epoch, in most situation, "
                        "we recommend 64 for cars196")
    parser.add_argument("--max_steps", type=int, default=8000,
                        help="The maximum step number")

    parser.add_argument("--Apply_HDML", type=bool, default=True,
                        help="Whether to apply hard-aware Negative Generation")
    parser.add_argument("--is_Training", type=bool, default=True,
                        help="Whether is training session or not")
    parser.add_argument("--Softmax_factor", type=float, default=1e+4,
                        help="The weight factor of softmax")
    parser.add_argument("--beta", type=float, default=1e+4,
                        help="The factor of negneg, 1e+4 for cars196")
    parser.add_argument("--lr_gen", type=float, default=1e-2,
                        help="1e-2 for cars196")
    parser.add_argument("--alpha", type=float, default=90,
                        help="The factor in the pulling function")
    parser.add_argument("--num_class", type=int, default=99,
                        help="Number of classes in dataset, 99 for cars, 101 for cub,"
                                             "11319 for products")
    parser.add_argument("--_lambda", type=float, default=0.5,
                        help="The trade_off between the two part of gen_loss, 0.5 for cars196")
    parser.add_argument("--s_lr", type=float, default=1e-3,
                        help="The learning rate of softmax trainer, 1e-3 for cars196")



    # Different losses need different method to create batches


    # Using GPU

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    return parser.parse_args()


# def main(_):
if __name__ == '__main__':


    #initialize
    args = get_arguments()
    if args.LossType == "Contrastive_Loss":
        method = "pair"
    elif args.LossType == "NpairLoss" or args.LossType == "AngularLoss" or args.LossType == "NCA_loss":
        method = "n_pairs_mc"
    elif args.LossType == "Triplet":
        method = 'triplet'
    else:
        method = "clustering"

    print("method : ",method)

    # Create the stream of datas from dataset
    streams = data_provider.get_streams(args.batch_size, args.dataSet, method, crop_size=args.default_image_size)
    stream_train, stream_train_eval, stream_test = streams

    _time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))  # import time이 FLAGS.py에 있음.
    LOGDIR = args.log_save_path + args.dataSet + '/' + args.LossType + '/' + _time + '/'



    # models

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    feature_net = inceptionv3.inception_v3(pretrained=True)
    net = PBML(num_classes=args.num_classes, net=feature_net)


    # extract feature


    # perform attention module

    # 1x1 convolution
