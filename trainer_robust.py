import argparse
import logging
import os
import copy
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
#from torchvision import transforms
from data_reader import H5DataLoader
import torch.nn.functional as F
from losses import BCE,GCE,SCE, BCE_Weighted
import pdb



def trainer_synapse(args, model, snapshot_path):
    #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    db_train = H5DataLoader(args.root_path)
    #HDF5Dataset('C:/ml/data', recursive=True, load_data=False, data_cache_size=4, transform=None)
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                           transform=transforms.Compose(
    #                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train.images)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    trainloader = db_train
    # worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if args.loss == 'SCE':
        ce_loss = SCE(args.num_classes, args.device, beta=args.beta)
    elif args.loss == 'GCE':
        ce_loss = GCE(args.num_classes, args.device, q=args.beta)
    elif args.loss == 'BCE':
        ce_loss = BCE(args.num_classes, args.device, beta=args.beta)
    else:
        ce_loss = nn.CrossEntropyLoss()


    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    # max_epoch = max_iterations // len(trainloader) + 1
    max_iterations = args.max_epochs * len(trainloader.images)/args.batch_size
    logging.info("{} iterations per epoch. {} max iterations ".format(
        len(trainloader.images)/args.batch_size, max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    robust_loss = copy.deepcopy(ce_loss)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(range(np.uint16(len(trainloader.images)/args.batch_size))):
            image_batch, label_batch = trainloader.next_batch(args.batch_size)
            image_batch = image_batch[:,:,:,None]
            image_batch = torch.from_numpy(
                image_batch).permute([0, 3, 1, 2]).to(torch.float)
            label_batch = torch.from_numpy(label_batch).permute([0, 3, 1, 2]).to(torch.float)
            #label_batch = label_batch[:,:,:,4]

            #image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            labs = torch.argmax(label_batch, dim=1, keepdim=False)
            
            class_weights = None
            if args.class_weight is not None:
                labs_onehot = torch.nn.functional.one_hot(torch.squeeze(labs), args.num_classes).float()
                tmp = labs_onehot.permute((0, 2, 3, 1))
                class_weights = 1 - (torch.sum(tmp.reshape((-1, 9)), axis=0) / (labs.shape[0]*labs.shape[1]*labs.shape[2]))
                
            # if class_weights.size(0) < 9:
            #     pdb.set_trace()
            if epoch_num < args.warmup:
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            else:
                ce_loss = robust_loss
                if args.class_weight is not None:
                    ce_loss = BCE_Weighted(args.num_classes, args.device, beta=args.beta, weights=class_weights)

            loss_ce = ce_loss(outputs, labs)
            loss_dice = dice_loss(outputs, labs, weight=class_weights, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' %
                         (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction1',
                                 outputs[1, ...] * 30, iter_num)
                labs1 = labs[1, ...].unsqueeze(0) * 30
                writer.add_image('train/GroundTruth1', labs1, iter_num)

        #save_interval = 50  # int(max_epoch/6)
        if 1: # save every epoch epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            if args.resume > 0:
                epoch_num = epoch_num + args.resume + 1
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:

            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
