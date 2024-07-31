import test as tnt_2
from settings import results_path
import copy
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, epoch_start
from settings import coefs
from settings import last_layer_optimizer_lr
from settings import warm_optimizer_lrs
from settings import joint_optimizer_lrs, joint_lr_step_size
from settings import train_dir, val_dir, train_push_dir, \
    train_batch_size, val_batch_size, train_push_batch_size
from settings import base_architecture, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, experiment_run
import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
import wandb
import shutil
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib.cm
from PIL import Image
import numpy as np
from lrp_resnet_canonized_poolconv_prototypes import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.init(project='X-ray-imbalance',  # entity='sga069'
           )

config = wandb.config

parser = argparse.ArgumentParser()
# python3 main.py -gpuid=0,1,2,3
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(
    os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(
    log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
val_dataset = datasets.ImageFolder(
    val_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('val set size: {0}'.format(len(val_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
# best_model_path = 'saved_models/resnet34/26-08-2021-Combined-orig/40_push0.6774.pth'
# ppnet.load_state_dict(torch.load(best_model_path), strict=False)

ppnet = ppnet.to(device)
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True
wandb.watch(ppnet, log="all")

# define optimizer
joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},  # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(
     ), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors,
            'lr': joint_optimizer_lrs['prototype_vectors']},
     ]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors,
         'lr': warm_optimizer_lrs['prototype_vectors']},
     ]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

last_layer_optimizer_specs = [
    {'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses

# number of training epochs, number of warm epochs, push start epoch, push epochs

# train the model
log('start training')
best_f1 = 0
best_epoch = 0
best_protopnet_heatmaps = []
best_prototype_images_path = []
best_model = []
for epoch in range(epoch_start, num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    f1 = tnt.test(model=ppnet_multi, dataloader=val_loader,
                  class_specific=class_specific, log=log)
    # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', f1=f1,
    #                             target_f1=0.70, log=log)

    if epoch >= push_start and epoch in push_epochs:
        protopnet_heatmaps_path, prototype_images_path = push.push_prototypes(
            # pytorch dataloader (must be unnormalized in [0,1])
            train_push_loader,
            # pytorch network with prototype_vectors
            prototype_network_parallel=ppnet_multi,
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            # if not None, prototypes will be saved here
            root_dir_for_saving_prototypes=img_dir,
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)

        # columns = ['Class', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
        # table = wandb.Table(data=saved_prototypes, columns=columns)
        # wandb.log({"prototypes": table})

        f1 = tnt.test(model=ppnet_multi, dataloader=val_loader,
                      class_specific=class_specific, log=log)
        # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', f1=f1,
        #                             target_f1=0.90, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                f1 = tnt.test(model=ppnet_multi, dataloader=val_loader,
                              class_specific=class_specific, log=log)
                if (f1 >= best_f1):
                    best_f1 = f1
                    best_epoch = epoch
                    best_protopnet_heatmaps = protopnet_heatmaps_path
                    best_prototype_images_path = prototype_images_path
                    best_model = copy.deepcopy(ppnet)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + 'push', f1=f1,
                                                target_f1=0.1, log=log)

    wandb.log({
        "Validation F1": f1,
        "Best F1": best_f1,
        "Best Epoch": best_epoch})


# Store prototype images
prototype_image_dir = results_path + '/prototype_images/prototype_images/'
makedir(prototype_image_dir)
n_prototypes = ppnet_multi.module.num_prototypes
for j in range(n_prototypes):
    shutil.copy(best_prototype_images_path[j],
                prototype_image_dir+str(j)+'_original.png')

# ProtoPNet heatmaps
protopnet_image_dir = results_path + '/ProtoPNet_heatmaps/'
makedir(protopnet_image_dir)
n_prototypes = ppnet_multi.module.num_prototypes
for j in range(n_prototypes):
    shutil.copy(best_protopnet_heatmaps[j],
                protopnet_image_dir+str(j)+'_ppnet.png')

# PRP
PRP_path = results_path + 'PRP_maps/'
makedir(PRP_path)
only_PRP_path = results_path+'PRP_maps/onlyPRP/'
n_prototypes = ppnet_multi.module.num_prototypes
makedir(only_PRP_path)
sim = run_prp(copy.deepcopy(best_model), results_path +
              '/prototype_images/', n_prototypes, PRP_path)


# Subplots
subplot_dir = results_path+'subplot/'
makedir(subplot_dir)

for j in range(n_prototypes):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(Image.open(prototype_image_dir+str(j)+'_original.png'))
    axs[0].axis('off')

    axs[1].imshow(Image.open(protopnet_image_dir+str(j)+'_ppnet.png'))
    axs[1].axis('off')

    axs[2].imshow(Image.open(PRP_path + str(j) + '-PRP.png'))
    axs[2].axis('off')

    plt.title(str(np.round(sim[j], 6)))
    plt.show()
    plt.tight_layout()
    plt.savefig(subplot_dir+str(j)+".png")


# Test f1 scores save
best_model_multi = torch.nn.DataParallel(best_model.to(device))


def cm(test_dir, filename, class_names):
    print(test_dir)
    test_batch_size = 100

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    print('test set size: {0}'.format(len(test_loader.dataset)))

    f1, pred, true = tnt_2.test(model=best_model_multi, dataloader=test_loader,
                                class_specific=class_specific, log=print)


logclose()
