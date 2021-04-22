"""
(Like YOLO v3) ResNet -> feature map -> FC(confidence/translation/scale/orientation) -> 3D bounding box
Reference:
    3D Bounding Box Estimation Using Deep Learning and Geometry(https://arxiv.org/abs/1612.00496)
    https://github.com/skhadem/3D-BoundingBox
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from top.model.backbone import resnet_fpn_backbone
from top.run.loss_util import *


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()


class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features

        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    nn.Softmax()
                )

        self.translation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )

        self.scale = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        confidence = self.confidence(x)
        translation = self.translation(x)
        scale = self.scale(x)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)

        return confidence, translation, scale, orientation


def train():
    
    # hyper parameters
    epochs = 100
    batch_size = 8
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '___'
    dataset = Dataset(train_path)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    backnone = resnet_fpn_backbone()
    model = Model(features=backnone.features).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, momentum=0.9)

    conf_loss_func = nn.CrossEntropyLoss().cuda()
    translation_loss_func = nn.MSELoss().cuda()
    scale_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(epochs):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['translation'].float().cuda()
            truth_dim = local_labels['scale'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            translation_loss = translation_loss_func(dim, truth_dim)
            scale_loss = translation_loss_func(dim, truth_dim)
            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss
            loss = alpha * translation_loss + loss_theta

            optim.zero_grad()
            loss.backward()
            optim.step()

            if passes % 10 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")


if __name__=='__main__':
    train()