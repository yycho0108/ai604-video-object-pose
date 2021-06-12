#!/usr/bin/env python3

import tqdm
from typing import List, Dict, Tuple, Hashable, Union
from dataclasses import dataclass
from simple_parsing import Serializable
import logging

import torch as th
from torchvision.transforms import Compose
from torch.utils.data._utils.collate import default_collate
from pytorch3d.transforms import quaternion_to_matrix

from top.data.transforms import (
    Normalize,
    InstancePadding,
    DenseMapsMobilePose,
    BoxPoints2D,
    SolveTranslation,
    DrawBoundingBoxFromKeypoints
)

from top.data.bbox_reg_util import CropObject
from top.data.load import (DatasetSettings, collate_cropped_img, get_loaders)
from top.data.schema import Schema
from top.train.trainer import Saver
from top.run.path_util import RunPath, get_latest_file
from top.run.app_util import update_settings
from top.run.torch_util import resolve_device

from top.model.bbox_3d import BoundingBoxRegressionModel
from top.run.draw_regressed_bbox import plot_regressed_3d_bbox

from matplotlib import pyplot as plt


def load_model():
    device = resolve_device('cuda')
    from bb_regression import AppSettings

    # Configure checkpoints + options
    ckpt = '/media/ssd/models/top/ckpt/step-117999.zip'
    opts = BoundingBoxRegressionModel.Settings()
    opts.load('/media/ssd/models/top/opts.yaml')

    # 1. load bbox regression model
    model = BoundingBoxRegressionModel(opts).to(device)
    logging.info(F'Loading checkpoint {ckpt} ...')
    Saver(model).load(ckpt)

    return model


def main():
    # data
    transform = Compose([CropObject(CropObject.Settings()), Normalize(
        Normalize.Settings(keys=(Schema.CROPPED_IMAGE,)))])
    _, test_loader = get_loaders(DatasetSettings(),
                                 th.device('cpu'),
                                 1,
                                 transform=transform,
                                 collate_fn=collate_cropped_img)
    # model
    device = th.device('cuda')
    model = load_model()
    model = model.to(device)
    model.eval()

    # translation solver?
    solve_translation = SolveTranslation()

    box_points = BoxPoints2D(th.device('cpu'), Schema.KEYPOINT_2D)
    draw_bbox = DrawBoundingBoxFromKeypoints(
        DrawBoundingBoxFromKeypoints.Settings())

    # eval
    for data in test_loader:
        # Skip occasional batches without any images.
        if Schema.CROPPED_IMAGE not in data:
            continue

        with th.no_grad():
            # run inference
            crop_img = data[Schema.CROPPED_IMAGE].view(-1, 3, 224, 224)
            dim, quat = model(crop_img.to(device))
            dim2, quat2 = data[Schema.SCALE], data[Schema.QUATERNION]
            logging.debug('D {} {}'.format(dim, dim2))
            logging.debug('Q {} {}'.format(quat, quat2))
            # trans = data[Schema.TRANSLATION]

            if False:
                dim = dim2
                quat = quat2
                R = quaternion_to_matrix(quat)

            R = quaternion_to_matrix(quat)

            input_image = data[Schema.IMAGE].detach().cpu()
            proj_matrix = (
                data[Schema.PROJECTION].detach().cpu().reshape(-1, 4, 4))

            # Solve translations.
            translations = []
            for i in range(len(proj_matrix)):
                box_i, box_j, box_h, box_w = data[Schema.BOX_2D][i]
                box_2d = th.as_tensor(
                    [box_i, box_j, box_i + box_h, box_j + box_w])
                box_2d = 2.0 * (box_2d - 0.5)
                args = {
                    # inputs from dataset
                    Schema.PROJECTION: proj_matrix[i],
                    Schema.BOX_2D: box_2d,
                    # inputs from network
                    Schema.ORIENTATION: R[i],
                    Schema.QUATERNION: quat[i],
                    Schema.SCALE: dim[i]
                }
                # Solve translation
                translation, _ = solve_translation(args)
                translations.append(translation)
            translations = th.as_tensor(translations, dtype=th.float32)

            if True:
                print('num instances = {}'.format(len(translations)))
                pred_data = {
                    Schema.IMAGE: data[Schema.IMAGE][0],
                    Schema.ORIENTATION: R.cpu(),
                    Schema.TRANSLATION: translations,
                    Schema.SCALE: dim.cpu(),
                    Schema.PROJECTION: proj_matrix[0],
                    Schema.INSTANCE_NUM: len(proj_matrix),
                }
                pred_data = box_points(pred_data)
                pred_data = draw_bbox(pred_data)
                image_with_box = pred_data['img_w_bbox']
            else:
                dimensions = dim.detach().cpu()
                quaternion = quat.detach().cpu()
                translations = translations.detach().cpu()

                #print(input_image.shape)
                #print(data[Schema.BOX_2D].shape)
                #print(proj_matrix.shape)
                #print(translations.shape)
                #print(dimensions.shape)
                #print(quaternion.shape)

                # draw box
                image_with_box = plot_regressed_3d_bbox(
                    input_image,
                    # keypoints_2d,
                    # data[Schema.BOX_2D],
                    data[Schema.KEYPOINT_2D],
                    proj_matrix,
                    dimensions,
                    quaternion,
                    translations)

            plt.clf()
            plt.imshow(image_with_box.permute(1, 2, 0))
            plt.pause(0.1)


if __name__ == '__main__':
    main()
