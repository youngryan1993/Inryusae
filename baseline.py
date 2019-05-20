import torch
import torch.nn as nn
import numpy as np
from datasets import data_provider
import argparse
import time
import os
import logging
import copy
from torch.utils.data import DataLoader
from datasets.dataset import CustomDataset
import warnings
from utils  import accuracy
from torchvision import transforms
from models import inceptionv3, inception_base
from lib import nn_Ops

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
    parser.add_argument("--batch_size", type=int, default=40,
                        help="batch size, 128 is recommended for cars196")
    parser.add_argument("--Regular_factor", type=float, default=5e-3,
                        help="weight decay factor, we recommend 5e-3 for cars196")
    parser.add_argument("--init_learning_rate", type=float, default=1e-2,
                        help="initial learning rate, we recommend 7e-5 for cars196")
    parser.add_argument("--default_image_size", type=int, default=299,
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
    parser.add_argument('--save-freq', dest='save_freq', default=1, type=int,
                        help='saving frequency of .ckpt models (default: 1)')
    parser.add_argument('--save-dir', dest='save_dir', default='./models',
                        help='saving directory of .ckpt models (default: ./models)')
    parser.add_argument('--epochs', dest='epochs', default=80, type=int,
                        help='number of epochs (default: 80)')
    parser.add_argument('--workers', dest='workers', default=16, type=int,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--verbose', dest='verbose', default=100, type=int,
                      help='show information for each <verbose> iterations (default: 100)')



    # Different losses need different method to create batches


    # Using GPU

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    return parser.parse_args()



def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    # feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']

    # metrics initialization
    batches = 0

    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float') # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    logging.info('Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        ##################################
        # Raw Image
        ##################################

        ############################################### y_pred, feature_matrix, attention_map = net(X)
        y_pred = net.forward(X)

        # loss
        batch_loss = loss(y_pred, y)  # + l2_loss(feature_matrix, feature_center[y])
        epoch_loss += batch_loss.item()

        # backward
        optimizer.zero_grad()
        batch_loss.backward(retain_graph=True)
        optimizer.step()

        # # Update Feature Center
        # feature_center[y] += beta * (feature_matrix.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            # print(type(epoch_acc))
            # print(accuracy(y_pred, y, topk=(1, 3, 5)))
            epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))


        # end of this batch
        batches += 1
        batch_end = time.time()
        if (i + 1) % verbose == 0:
            logging.info(
                '\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f),  Time %3.2f' %
                (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches, epoch_acc[2] / batches, batch_end - batch_start))

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,},
            # 'feature_center': feature_center.cpu()},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info(
        'Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f),  Time %3.2f' %
        (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2],
         # epoch_loss[1], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
         # epoch_loss[2], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
         end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']

    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            batch_start = time.time()

            # obtain data
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            ##################################
            # Raw Image
            ##################################
            y_pred = net.forward(X)


            # loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss.item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))

            # end of this batch
            batches += 1
            batch_end = time.time()
            if (i + 1) % verbose == 0:
                logging.info('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                             (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches,
                              epoch_acc[2] / batches, batch_end - batch_start))

    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
    logging.info('')

    return epoch_loss


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
    # streams = data_provider.get_streams(args.batch_size, args.dataSet, method, crop_size=args.default_image_size)
    # stream_train, stream_train_eval, stream_test = streams
    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")
    _time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))  # import time이 FLAGS.py에 있음.
    LOGDIR = args.log_save_path + args.dataSet + '/' + args.LossType + '/' + _time + '/'

    # epoch_iterator = stream_train.get_epoch_iterator()
    # test_iterator = stream_train.get_epoch_iterator()

    # bp_epoch = args.init_batch_per_epoch
    step = 0
    bp_epoch = 500
    image_size = (299, 299)
    num_classes = 1000
    # num_attentions = 32
    start_epoch = 0
    # models

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = CustomDataset(phase='train', shape=image_size), \
                                      CustomDataset(phase='val'  , shape=image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=args.batch_size * 4, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)



    feature_net = inceptionv3.inception_v3(pretrained=True)
    nets = inception_base.base(feature_net).cuda()
    # nets = torch.load('./ckpt1')
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(nets.parameters(), lr = args.init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    idx = 0
    best = 0

    for epoch in range(start_epoch, args.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=nets,
              # feature_center=feature_center,
              loss=loss,
              optimizer=optimizer,
              save_freq=args.save_freq,
              save_dir=args.save_dir,
              verbose=args.verbose)
        val_loss = validate(data_loader=validate_loader,
                            net=nets,
                            loss=loss,
                            verbose=args.verbose)
        scheduler.step()
