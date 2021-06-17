#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK
"""Example Evaluation script for Objectron dataset.

It reads a tfrecord, runs evaluation, and outputs a summary report with name
specified in report_file argument. When adopting this for your own model, you
have to implement the Evaluator.predict() function, which takes an image and produces
a 3D bounding box.

Example:
    python3 -m objectron.dataset.eval --eval_data=.../chair_test* --report_file=.../report.txt
"""

import math
import os
import warnings
import glob
import numpy as np
import tqdm
from typing import List, Dict, Tuple, Hashable, Union
from dataclasses import dataclass
from simple_parsing import Serializable
import logging

import torch as th
from torchvision.transforms import Compose
from torch.utils.data._utils.collate import default_collate
import torch.autograd.profiler as profiler
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

from top.data.transforms import (
    Normalize,
    InstancePadding,
    DenseMapsMobilePose
)
from top.data.transforms.keypoint import (
    BoxPoints2D
)
from top.data.transforms.bounding_box import (
    SolveTranslation
)
from top.data.bbox_reg_util import CropObject
from top.data.load import (DatasetSettings, collate_cropped_img, get_loaders)
from top.data.schema import Schema
from top.train.trainer import Saver
from top.run.metrics_util import (
    IoU, Box, AveragePrecision, HitMiss)
from top.run.path_util import RunPath, get_latest_file
from top.run.app_util import update_settings
from top.run.torch_util import resolve_device

from test_keypoint_decoder import GroundTruthDecoder
from top.model.bbox_3d import BoundingBoxRegressionModel


def safe_divide(i1, i2, eps: float = 1e-6):
    divisor = float(i2) if i2 > 0 else eps
    return i1 / divisor


@dataclass
class AppSettings(Serializable):
    dataset: DatasetSettings = DatasetSettings()

    max_pixel_error: float = 20.
    max_azimuth_error: float = 30.  # in degrees
    max_polar_error: float = 20.  # in degrees
    max_distance: float = 1.0  # In meters
    num_bins: float = 21  # ??

    # threshold for the visibility
    vis_thresh: float = 0.1

    # FIXME(ycho): Restore functional `collate_fn`.
    batch_size: int = 1
    # Max number of samples from the test dataset to evaluate.
    max_num: int = -1
    report_file: str = '/tmp/report.txt'
    device: str = ''
    profile: bool = False


class FormatLabel:
    """`torch.Transform` version of `ObjectronParser`.

    NOTE(ycho): Despite how it looks, this transform does NOT override inputs,
    since input keys are `Schema`s and Settings are `str`s, (by default).
    """

    @dataclass
    class Settings(Serializable):
        key_2d: str = '2d_instance'
        key_3d: str = '3d_instance'
        visibility: str = 'visibility'

    def __init__(self, opts: Settings, vis_thresh: float):
        self.opts = opts
        self.vis_thresh = vis_thresh

    def __call__(self, inputs: Dict[Schema,
                                    th.Tensor]) -> Dict[Schema, th.Tensor]:
        outputs = inputs.copy()

        visibilities = inputs[Schema.VISIBILITY].reshape(-1)
        outputs[self.opts.visibility] = visibilities
        visible_mask = (visibilities > self.vis_thresh)

        points_2d = inputs[Schema.KEYPOINT_2D]
        points_2d = (points_2d).reshape((-1, 9, 3))[..., :2]
        outputs[self.opts.key_2d] = points_2d[visible_mask]

        if Schema.KEYPOINT_3D in inputs:
            points_3d = inputs[Schema.KEYPOINT_3D]
            points_3d = (points_3d).reshape((-1, 9, 3))
            outputs[self.opts.key_3d] = points_3d[visible_mask]

        return outputs


class Evaluator(object):
    """Class for evaluating the Objectron's model."""

    def __init__(self, opts: AppSettings,
                 model: th.nn.Module):
        self.opts = opts
        self.model = model

        self._error_2d = 0.
        self._matched = 0

        self._iou_3d = 0.
        self._azimuth_error = 0.
        self._polar_error = 0.

        self._iou_thresholds = np.linspace(0.0, 1., num=opts.num_bins)
        self._pixel_thresholds = np.linspace(
            0.0, opts.max_pixel_error, num=opts.num_bins)
        self._azimuth_thresholds = np.linspace(
            0.0, opts.max_azimuth_error, num=opts.num_bins)
        self._polar_thresholds = np.linspace(
            0.0, opts.max_polar_error, num=opts.num_bins)
        self._add_thresholds = np.linspace(
            0.0, opts.max_distance, num=opts.num_bins)
        self._adds_thresholds = np.linspace(
            0.0, opts.max_distance, num=opts.num_bins)

        self._iou_ap = AveragePrecision(opts.num_bins)
        self._pixel_ap = AveragePrecision(opts.num_bins)
        self._azimuth_ap = AveragePrecision(opts.num_bins)
        self._polar_ap = AveragePrecision(opts.num_bins)
        self._add_ap = AveragePrecision(opts.num_bins)
        self._adds_ap = AveragePrecision(opts.num_bins)

    def predict(self, images, batch_size):
        """Implement your own function/model to predict the box's 2D and 3D
        keypoint from the input images. Note that the predicted 3D bounding
        boxes are correct upto an scale. You can use the ground planes to re-
        scale your boxes if necessary.

        Returns:
            A list of list of boxes for objects in images in the batch. Each box is
            a tuple of (point_2d, point_3d) that includes the predicted 2D and 3D vertices.
        """
        # TODO(ycho): preprocess ... [usually, normalize images]
        prev_mode = self.model.training
        self.model.eval()
        with th.no_grad():
            output = self.model(images)
        self.model.train(prev_mode)
        # TODO(ycho): postprocess ... [usually, decode boxes]
        return output

    def evaluate(self, batch: Dict[Hashable, th.Tensor]):
        """Evaluates a batch of serialized tf.Example protos."""
        opts = self.opts

        # Same as `predict()`, but not on just images for models
        # that require more information (GroundTruthDecoder, BboxWrapper)
        prev_mode = self.model.training
        self.model.eval()
        with th.no_grad():
            results = self.model(batch)
        self.model.train(prev_mode)

        if results is None:
            return

        # NOTE(ycho): For BboxWrapper we do not pass in collated input.
        if isinstance(self.model, BboxWrapper):
            batch = default_collate(batch)

        # NOTE(ycho): We need results as a list of 3d boxes. Yep!
        # Since we already parsed everything, no need to be re-parse
        labels = {k: (v.detach().cpu().numpy() if isinstance(
            v, th.Tensor) else v) for (k, v) in batch.items()}

        # Creating some fake results for testing as well as example of what the
        # the results should look like.
        # results = []
        # for label in labels:
        #   instances = label['2d_instance']
        #   instances_3d = label['3d_instance']
        #   boxes = []
        #   for i in range(len(instances)):
        #       point_2d = np.copy(instances[i])
        #       point_3d = np.copy(instances_3d[i])
        #       for j in range(9):
        #           # Translating the box in 3D, this will have a large impact on 3D IoU.
        #           point_3d[j] += np.array([0.01, 0.02, 0.5])
        #       boxes.append((point_2d, point_3d))
        #   results.append(boxes)

        for i_batch, boxes in enumerate(results):
            instances = labels['2d_instance'][i_batch]
            instances_3d = labels['3d_instance'][i_batch]
            visibilities = labels['visibility'][i_batch]

            # NOTE(ycho): Counting the number of visible instances
            # throughout the entire batch.
            num_instances = 0
            for instance, instance_3d, visibility in zip(
                    instances, instances_3d, visibilities):
                if (visibility > self.opts.vis_thresh and self._is_visible(
                        instance[0]) and instance_3d[0, 2] < 0):
                    num_instances += 1

            # We don't have negative examples in evaluation.
            if num_instances == 0:
                continue

            iou_hit_miss = HitMiss(self._iou_thresholds)
            azimuth_hit_miss = HitMiss(self._azimuth_thresholds)
            polar_hit_miss = HitMiss(self._polar_thresholds)
            pixel_hit_miss = HitMiss(self._pixel_thresholds)
            add_hit_miss = HitMiss(self._add_thresholds)
            adds_hit_miss = HitMiss(self._adds_thresholds)

            num_matched = 0
            for box in boxes:
                box_point_2d, box_point_3d = box
                index = self.match_box(box_point_2d, instances, visibilities)
                if index >= 0:
                    num_matched += 1
                    pixel_error = self.evaluate_2d(
                        box_point_2d, instances[index])
                    azimuth_error, polar_error, iou, add, adds = self.evaluate_3d(
                        box_point_3d, instances_3d[index])
                else:
                    pixel_error = opts.max_pixel_error
                    azimuth_error = opts.max_azimuth_error
                    polar_error = opts.max_polar_error
                    iou = 0.
                    add = opts.max_distance
                    adds = opts.max_distance

                iou_hit_miss.record_hit_miss(iou)
                add_hit_miss.record_hit_miss(add, greater=False)
                adds_hit_miss.record_hit_miss(adds, greater=False)
                pixel_hit_miss.record_hit_miss(pixel_error, greater=False)
                azimuth_hit_miss.record_hit_miss(azimuth_error, greater=False)
                polar_hit_miss.record_hit_miss(polar_error, greater=False)

            self._iou_ap.append(iou_hit_miss, len(instances))
            self._pixel_ap.append(pixel_hit_miss, len(instances))
            self._azimuth_ap.append(azimuth_hit_miss, len(instances))
            self._polar_ap.append(polar_hit_miss, len(instances))
            self._add_ap.append(add_hit_miss, len(instances))
            self._adds_ap.append(adds_hit_miss, len(instances))
            self._matched += num_matched

    def evaluate_2d(self, box, instance):
        """Evaluates a pair of 2D projections of 3D boxes.

        It computes the mean normalized distances of eight vertices of a box.

        Args:
            box: A 9*2 array of a predicted box.
            instance: A 9*2 array of an annotated box.

        Returns:
            Pixel error
        """
        error = np.mean(np.linalg.norm(box[1:] - instance[1:], axis=1))
        self._error_2d += error
        return error

    def evaluate_3d(self, box_point_3d, instance):
        """Evaluates a box in 3D.

        It computes metrics of view angle and 3D IoU.

        Args:
            box: A predicted box.
            instance: A 9*3 array of an annotated box, in metric level.

        Returns:
            A tuple containing the azimuth error, polar error, 3D IoU (float),
            average distance error, and average symmetric distance error.
        """
        azimuth_error, polar_error = self.evaluate_viewpoint(
            box_point_3d, instance)
        avg_distance, avg_sym_distance = self.compute_average_distance(
            box_point_3d,
            instance)
        iou = self.evaluate_iou(box_point_3d, instance)
        return azimuth_error, polar_error, iou, avg_distance, avg_sym_distance

    def compute_scale(self, box, plane):
        """Computes scale of the given box sitting on the plane."""
        center, normal = plane
        vertex_dots = [np.dot(vertex, normal) for vertex in box[1:]]
        vertex_dots = np.sort(vertex_dots)
        center_dot = np.dot(center, normal)
        scales = center_dot / vertex_dots[:4]
        return np.mean(scales)

    def compute_ray(self, box):
        """Computes a ray from camera to box centroid in box frame.

        For vertex in camera frame V^c, and object unit frame V^o, we have
            R * Vc + T = S * Vo,
        where S is a 3*3 diagonal matrix, which scales the unit box to its real size.

        In fact, the camera coordinates we get have scale ambiguity. That is, we have
            Vc' = 1/beta * Vc, and S' = 1/beta * S
        where beta is unknown. Since all box vertices should have negative Z values,
        we can assume beta is always positive.

        To update the equation,
            R * beta * Vc' + T = beta * S' * Vo.

        To simplify,
            R * Vc' + T' = S' * Vo,
        where Vc', S', and Vo are known. The problem is to compute
            T' = 1/beta * T,
        which is a point with scale ambiguity. It forms a ray from camera to the
        centroid of the box.

        By using homogeneous coordinates, we have
            M * Vc'_h = (S' * Vo)_h,
        where M = [R|T'] is a 4*4 transformation matrix.

        To solve M, we have
            M = ((S' * Vo)_h * Vc'_h^T) * (Vc'_h * Vc'_h^T)_inv.
        And T' = M[:3, 3:].

        Args:
            box: A 9*3 array of a 3D bounding box.

        Returns:
            A ray represented as [x, y, z].
        """
        if box[0, -1] > 0:
            warnings.warn('Box should have negative Z values.')

        size_x = np.linalg.norm(box[5] - box[1])
        size_y = np.linalg.norm(box[3] - box[1])
        size_z = np.linalg.norm(box[2] - box[1])
        size = np.asarray([size_x, size_y, size_z])
        box_o = Box.UNIT_BOX * size
        box_oh = np.ones((4, 9))
        box_oh[:3] = np.transpose(box_o)

        box_ch = np.ones((4, 9))
        box_ch[:3] = np.transpose(box)
        box_cht = np.transpose(box_ch)

        box_oct = np.matmul(box_oh, box_cht)
        box_cct_inv = np.linalg.inv(np.matmul(box_ch, box_cht))
        transform = np.matmul(box_oct, box_cct_inv)
        return transform[:3, 3:].reshape((3))

    def compute_average_distance(self, box, instance):
        """Computes Average Distance (ADD) metric."""
        add_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            delta = np.linalg.norm(box[i, :] - instance[i, :])
            add_distance += delta
        add_distance /= Box.NUM_KEYPOINTS

        # Computes the symmetric version of the average distance metric.
        # From PoseCNN https://arxiv.org/abs/1711.00199
        # For each keypoint in predicttion, search for the point in ground truth
        # that minimizes the distance between the two.
        add_sym_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            # Find nearest vertex in instance
            distance = np.linalg.norm(box[i, :] - instance[0, :])
            for j in range(Box.NUM_KEYPOINTS):
                d = np.linalg.norm(box[i, :] - instance[j, :])
                if d < distance:
                    distance = d
            add_sym_distance += distance
        add_sym_distance /= Box.NUM_KEYPOINTS

        return add_distance, add_sym_distance

    def compute_viewpoint(self, box):
        """Computes viewpoint of a 3D bounding box.

        We use the definition of polar angles in spherical coordinates
        (http://mathworld.wolfram.com/PolarAngle.html), expect that the
        frame is rotated such that Y-axis is up, and Z-axis is out of screen.

        Args:
            box: A 9*3 array of a 3D bounding box.

        Returns:
            Two polar angles (azimuth and elevation) in degrees. The range is between
            -180 and 180.
        """
        x, y, z = self.compute_ray(box)
        theta = math.degrees(math.atan2(z, x))
        phi = math.degrees(math.atan2(y, math.hypot(x, z)))
        return theta, phi

    def evaluate_viewpoint(self, box, instance):
        """Evaluates a 3D box by viewpoint.

        Args:
            box: A 9*3 array of a predicted box.
            instance: A 9*3 array of an annotated box, in metric level.

        Returns:
            Two viewpoint angle errors.
        """
        predicted_azimuth, predicted_polar = self.compute_viewpoint(box)
        gt_azimuth, gt_polar = self.compute_viewpoint(instance)

        polar_error = abs(predicted_polar - gt_polar)
        # Azimuth is from (-180,180) and a spherical angle so angles -180 and 180
        # are equal. E.g. the azimuth error for -179 and 180 degrees is 1'.
        azimuth_error = abs(predicted_azimuth - gt_azimuth)
        if azimuth_error > 180:
            azimuth_error = 360 - azimuth_error

        self._azimuth_error += azimuth_error
        self._polar_error += polar_error
        return azimuth_error, polar_error

    def evaluate_rotation(self, box, instance):
        """Evaluates rotation of a 3D box.

        1. The L2 norm of rotation angles
        2. The rotation angle computed from rotation matrices
                    trace(R_1^T R_2) = 1 + 2 cos(theta)
                    theta = arccos((trace(R_1^T R_2) - 1) / 2)

        3. The rotation angle computed from quaternions. Similar to the above,
             except instead of computing the trace, we compute the dot product of two
             quaternion.
                 theta = 2 * arccos(| p.q |)
             Note the distance between quaternions is not the same as distance between
             rotations.

        4. Rotation distance from "3D Bounding box estimation using deep learning
             and geometry""
                     d(R1, R2) = || log(R_1^T R_2) ||_F / sqrt(2)

        Args:
            box: A 9*3 array of a predicted box.
            instance: A 9*3 array of an annotated box, in metric level.

        Returns:
            Magnitude of the rotation angle difference between the box and instance.
        """
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        gt_rotation_inverse = np.linalg.inv(annotation.rotation)
        rotation_error = np.matmul(prediction.rotation, gt_rotation_inverse)

        error_angles = np.array(
            rotation_util.from_dcm(rotation_error).as_euler('zxy'))
        abs_error_angles = np.absolute(error_angles)
        abs_error_angles = np.minimum(
            abs_error_angles, np.absolute(
                math.pi * np.ones(3) - abs_error_angles))
        error = np.linalg.norm(abs_error_angles)

        # Compute the error as the angle between the two rotation
        rotation_error_trace = abs(np.matrix.trace(rotation_error))
        angular_distance = math.acos((rotation_error_trace - 1.) / 2.)

        # angle = 2 * acos(|q1.q2|)
        box_quat = np.array(rotation_util.from_dcm(
            prediction.rotation).as_quat())
        gt_quat = np.array(rotation_util.from_dcm(
            annotation.rotation).as_quat())
        quat_distance = 2 * math.acos(np.dot(box_quat, gt_quat))

        # The rotation measure from "3D Bounding box estimation using deep learning
        # and geometry"
        rotation_error_log = scipy.linalg.logm(rotation_error)
        rotation_error_frob_norm = np.linalg.norm(
            rotation_error_log, ord='fro')
        rotation_distance = rotation_error_frob_norm / 1.4142

        return (error, quat_distance, angular_distance, rotation_distance)

    def evaluate_iou(self, box, instance):
        """Evaluates a 3D box by 3D IoU.

        It computes 3D IoU of predicted and annotated boxes.

        Args:
            box: A 9*3 array of a predicted box.
            instance: A 9*3 array of an annotated box, in metric level.

        Returns:
            3D Intersection over Union (float)
        """
        # Computes 3D IoU of the two boxes.
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        iou = IoU(prediction, annotation)
        iou_result = iou.iou()
        self._iou_3d += iou_result
        return iou_result

    def match_box(self, box, instances, visibilities):
        """Matches a detected box with annotated instances.

        For a predicted box, finds the nearest annotation in instances. This means
        we always assume a match for a prediction. If the nearest annotation is
        below the visibility threshold, the match can be skipped.

        Args:
            box: A 9*2 array of a predicted box.
            instances: A ?*9*2 array of annotated instances. Each instance is a 9*2
                array.
            visibilities: An array of the visibilities of the instances.

        Returns:
            Index of the matched instance; otherwise -1.
        """
        norms = np.linalg.norm(instances[:, 1:, :] - box[1:, :], axis=(1, 2))
        i_min = np.argmin(norms)
        if visibilities[i_min] < self.opts.vis_thresh:
            return -1
        return i_min

    def write_report(self):
        """Writes a report of the evaluation."""
        opts = self.opts

        def report_array(f, label, array):
            f.write(label)
            for val in array:
                f.write('{:.4f},\t'.format(val))
            f.write('\n')

        report_file = opts.report_file

        with open(report_file, 'w') as f:
            f.write('Mean Error 2D: {}\n'.format(
                    safe_divide(self._error_2d, self._matched)))
            f.write('Mean 3D IoU: {}\n'.format(
                    safe_divide(self._iou_3d, self._matched)))
            f.write('Mean Azimuth Error: {}\n'.format(
                    safe_divide(self._azimuth_error, self._matched)))
            f.write('Mean Polar Error: {}\n'.format(
                    safe_divide(self._polar_error, self._matched)))

            f.write('\n')
            f.write('IoU Thresholds: ')
            for threshold in self._iou_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @3D IoU     : ', self._iou_ap.aps)

            f.write('\n')
            f.write('2D Thresholds : ')
            for threshold in self._pixel_thresholds:
                f.write('{:.4f},\t'.format(threshold * 0.1))
            f.write('\n')
            report_array(f, 'AP @2D Pixel   : ', self._pixel_ap.aps)
            f.write('\n')

            f.write('Azimuth Thresh: ')
            for threshold in self._azimuth_thresholds:
                f.write('{:.4f},\t'.format(threshold * 0.1))
            f.write('\n')
            report_array(f, 'AP @Azimuth     : ', self._azimuth_ap.aps)
            f.write('\n')

            f.write('Polar Thresh   : ')
            for threshold in self._polar_thresholds:
                f.write('{:.4f},\t'.format(threshold * 0.1))
            f.write('\n')
            report_array(f, 'AP @Polar       : ', self._polar_ap.aps)
            f.write('\n')

            f.write('ADD Thresh     : ')
            for threshold in self._add_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @ADD             : ', self._add_ap.aps)
            f.write('\n')

            f.write('ADDS Thresh     : ')
            for threshold in self._adds_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @ADDS           : ', self._adds_ap.aps)

    def finalize(self):
        """Computes average precision curves."""
        self._iou_ap.compute_ap_curve()
        self._pixel_ap.compute_ap_curve()
        self._azimuth_ap.compute_ap_curve()
        self._polar_ap.compute_ap_curve()
        self._add_ap.compute_ap_curve()
        self._adds_ap.compute_ap_curve()

    def _is_visible(self, point):
        """Determines if a 2D point is visible."""
        return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1


class BboxWrapper(th.nn.Module):
    def __init__(self, model: th.nn.Module, device: th.device):
        super().__init__()
        self.model = model
        self.device = device

        self.transform = Compose([
            CropObject(CropObject.Settings()),
            Normalize(Normalize.Settings(keys=(Schema.CROPPED_IMAGE,)))
        ])
        self.solve_translation = SolveTranslation()
        self.box_points = BoxPoints2D(device,
                                      key_out=Schema.KEYPOINT_2D,
                                      key_out_3d=Schema.KEYPOINT_3D)

    def forward(self, inputs: List[Dict[Hashable, th.Tensor]]
                ) -> List[List[Tuple[th.Tensor, th.Tensor]]]:
        inputs0 = inputs.copy()

        # NOTE(ycho): custom transform + crop-aware collation.
        inputs = [self.transform(x) for x in inputs]
        inputs = collate_cropped_img(inputs)
        if (Schema.CROPPED_IMAGE not in inputs or
                len(inputs[Schema.CROPPED_IMAGE]) <= 0):
            return None

        # if we use the crop-aware collation, (batch_idx, instance_idx)
        indices = inputs[Schema.INDEX]
        dim, quat = self.model(inputs[Schema.CROPPED_IMAGE].to(
            self.device))

        # NOTE(ycho): len(image) appropriated for batch_size
        batch_size = len(inputs0)
        outputs = [[] for _ in range(batch_size)]

        for i, (ii, s, q) in enumerate(zip(indices, dim, quat)):
            batch_index, instance_index = ii
            P = inputs0[batch_index][Schema.PROJECTION].reshape(4, 4)
            R = quaternion_to_matrix(q[None])[0]

            #R2 = inputs0[batch_index][
            #    Schema.ORIENTATION][instance_index].reshape(
            #    3, 3)
            #q2 = matrix_to_quaternion(R2[None])[0]
            #s2 = inputs0[batch_index][Schema.SCALE][instance_index]

            # Fix BOX_2D convention.
            box_i, box_j, box_h, box_w = inputs[Schema.BOX_2D][i]
            box_2d = th.as_tensor([box_i, box_j, box_i + box_h, box_j + box_w])
            box_2d = 2.0 * (box_2d - 0.5)

            # Solve translation
            translation, _ = self.solve_translation({
                # inputs from dataset
                Schema.PROJECTION: P,
                Schema.BOX_2D: box_2d,
                # inputs from network
                Schema.ORIENTATION: R,
                Schema.QUATERNION: q,
                Schema.SCALE: s
                # inputs from dataset (ground-truth)
                # Schema.ORIENTATION: R2,
                # Schema.QUATERNION: q2,
                # Schema.SCALE: s2
            })

            translation = th.as_tensor(translation, device=R.device)
            if translation[-1] > 0:
                translation *= -1.0
            #print('tr-solve')
            #print(translation)
            #print('tr-gt')
            #print(inputs0[batch_index][Schema.TRANSLATION][instance_index])

            # Convert to box-points
            box_out = self.box_points({
                Schema.ORIENTATION: R.to(self.device),
                Schema.TRANSLATION: translation.to(self.device),
                Schema.SCALE: s.to(self.device),
                Schema.PROJECTION: P.to(self.device),
                Schema.INSTANCE_NUM: 1
            })

            entry = (
                box_out[Schema.KEYPOINT_2D][0, ..., :2].detach().cpu().numpy(),
                box_out[Schema.KEYPOINT_3D][0].detach().cpu().numpy()
            )
            outputs[batch_index].append(entry)
        return outputs


def load_model():
    if False:
        # Instantiation ...
        device = resolve_device('cuda')
        opts = KeypointNetwork2D.Settings()
        # opts.load...()
        model = KeypointNetwork2D(opts).to(device)

        # Load checkpoint.
        ckpt_file = get_latest_file('/tmp/ai604-kpt/run-031/ckpt/')
        Saver(model, None).load(ckpt_file)
    elif True:
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

        # 2. *but* use the wrapper to convert to/from raw inputs/outputs
        # required for the `BoundingBoxRegressionModel`.
        return BboxWrapper(model, device)
    else:
        return GroundTruthDecoder()

    return model


def eval_main(opts: AppSettings):

    model = load_model()
    evaluator = Evaluator(opts, model)

    if isinstance(model, GroundTruthDecoder):
        # for Keypoints (+ ground-truth kpt-style decoder)
        transform = Compose([
            DenseMapsMobilePose(DenseMapsMobilePose.Settings(),
                                th.device('cpu')),
            Normalize(Normalize.Settings()),
            InstancePadding(InstancePadding.Settings()),
            # TODO(ycho): Does the order between padding<->label matter here?
            FormatLabel(FormatLabel.Settings(), opts.vis_thresh),
        ])
        collate_fn = None
    elif isinstance(model, BboxWrapper):
        # for Bounding Box
        transform = Compose([
            # NOTE(ycho): `FormatLabel` must be applied prior
            # to `CropObject` since it modifies the requisite tensors.
            FormatLabel(FormatLabel.Settings(), opts.vis_thresh),
            # CropObject(CropObject.Settings()),
            # Normalize(Normalize.Settings(keys=(Schema.CROPPED_IMAGE,)))
        ])

        # NOTE(ycho): passthrough collation;
        # actual collation will be handled independently.
        def collate_fn(data):
            return data

    _, test_loader = get_loaders(opts.dataset,
                                 th.device('cpu'),
                                 opts.batch_size,
                                 transform=transform,
                                 collate_fn=collate_fn)

    # Run evaluation ...
    try:
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            evaluator.evaluate(data)
            if (opts.max_num >= 0) and (i >= opts.max_num):
                break
    except KeyboardInterrupt:
        pass
    finally:
        evaluator.finalize()
        evaluator.write_report()


def main():
    logging.captureWarnings(True)
    opts = AppSettings()
    opts = update_settings(opts)
    if opts.profile:
        try:
            with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                eval_main(opts)
        finally:
            print('tracing...')
            print(prof.key_averages().table(
                sort_by='cpu_time_total',
                row_limit=16))
            prof.export_chrome_trace("/tmp/trace.json")
    else:
        eval_main(opts)


if __name__ == '__main__':
    main()
