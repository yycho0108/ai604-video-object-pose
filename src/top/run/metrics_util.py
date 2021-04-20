"""
Util classes for computing evaluation metrics.(HitMiss, Accuracy, AveragePrecision, IoU)
Reference: https://github.com/google-research-datasets/Objectron/blob/master/objectron/dataset/metrics.py
"""

import numpy as np
import scipy.spatial as sp
import src.top.run.box_generator as Box


class HitMiss(object):
    """Class for recording hits and misses of detection results."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.size = thresholds.shape[0]
        self.hit = np.zeros(self.size, dtype=np.float)
        self.miss = np.zeros(self.size, dtype=np.float)

    def reset(self):
        self.hit = np.zeros(self.size, dtype=np.float)
        self.miss = np.zeros(self.size, dtype=np.float)

    def record_hit_miss(self, metric, greater=True):
        """Records the hit or miss for the object based on the metric threshold."""
        # NOTE(Jiyong): greater True -> find greater than threshold / False -> find worse than threshold

        for i in range(self.size):
            threshold = self.thresholds[i]
            hit = (greater and metric >= threshold) or ((not greater) and metric <= threshold)
            if hit:
                self.hit[i] += 1
            else:
                self.miss[i] += 1


class Accuracy(object):
    """Class for accuracy metric."""

    def __init__(self):
        self._errors = []
        self.acc = []

    def add_error(self, error):
        """Adds an error."""
        self._errors.append(error)

    def compute_accuracy(self, thresh=0.1):
        """Computes accuracy for a given threshold."""
        if not self._errors:
            return 0
        return len(np.where(np.array(self._errors) <= thresh)[0]) * 100. / len(self._errors)


class AveragePrecision(object):
    """Class for computing average precision."""

    def __init__(self, size):
        self.size = size
        self.aps = np.zeros(size)
        self.true_positive = []
        self.false_positive = []
        for _ in range(size):
            self.true_positive.append([])
            self.false_positive.append([])
        self._total_instances = 0.

    def append(self, hit_miss, num_instances):
        for i in range(self.size):
            self.true_positive[i].append(hit_miss.hit[i])
            self.false_positive[i].append(hit_miss.miss[i])
        self._total_instances += num_instances

    def compute_ap(self, recall, precision):
        """
        Calculates the AP given the recall and precision array.
        The reference implementation is from Pascal VOC 2012 eval script.
        First we filter the precision recall rate so precision would be monotonically decreasing.
        Next, we compute the average precision by numerically integrating the precision-recall curve.
        Args:
            recall: Recall list
            precision: Precision list
        Returns:
            Average precision
        """
        recall = np.insert(recall, 0, [0.])
        recall = np.append(recall, [1.])
        precision = np.insert(precision, 0, [0.])
        precision = np.append(precision, [0.])
        monotonic_precision = precision.copy()
        
        # Make the precision monotonically decreasing.
        for i in range(len(monotonic_precision) - 2, -1, -1):
            monotonic_precision[i] = max(monotonic_precision[i],
                                         monotonic_precision[i + 1])

        recall_changes = []
        for i in range(1, len(recall)):
            if recall[i] != recall[i - 1]:
                recall_changes.append(i)

        # Compute the average precision by integrating the recall curve.
        ap = 0.0
        for step in recall_changes:
            delta_recall = recall[step] - recall[step - 1]
            ap += delta_recall * monotonic_precision[step]
        return ap

    def compute_ap_curve(self):
        """Computes the precision/recall curve."""
        if self._total_instances == 0:
            raise ValueError('No instances in the computation.')

        for i in range(self.size):
            tp, fp = self.true_positive[i], self.false_positive[i]
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            tp_fp = tp + fp
            recall = tp / self._total_instances
            precision = np.divide(tp, tp_fp, out=np.zeros_like(tp), where=tp_fp != 0)
            self.aps[i] = self.compute_ap(recall, precision)


# Global variables for IoU
_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1

class IoU(object):
    """General Intersection Over Union cost for Oriented 3D bounding boxes."""

    def __init__(self, box1, box2):
        self._box1 = box1
        self._box2 = box2
        self._intersection_points = []

    def iou(self):
        """Computes the exact IoU using Sutherland-Hodgman algorithm."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)
        if self._intersection_points:
            intersection_volume = sp.ConvexHull(self._intersection_points).volume
            box1_volume = self._box1.volume
            box2_volume = self._box2.volume
            union_volume = box1_volume + box2_volume - intersection_volume
            return intersection_volume / union_volume
        else:
            return 0.

    def _compute_intersection_points(self, box_src, box_template):
        """Computes the intersection of two boxes."""
        # Transform the source box to be axis-aligned
        inv_transform = np.linalg.inv(box_src.transformation)
        box_src_axis_aligned = box_src.apply_transformation(inv_transform)
        template_in_src_coord = box_template.apply_transformation(inv_transform)
        for face in range(len(Box.FACES)):
            indices = Box.FACES[face, :]
            poly = [template_in_src_coord.vertices[indices[i], :] for i in range(4)]
            clip = self.intersect_box_poly(box_src_axis_aligned, poly)
            for point in clip:
                # Transform the intersection point back to the world coordinate
                point_w = np.matmul(box_src.rotation, point) + box_src.translation
                self._intersection_points.append(point_w)

        for point_id in range(Box.NUM_KEYPOINTS):
            v = template_in_src_coord.vertices[point_id, :]
            if box_src_axis_aligned.inside(v):
                point_w = np.matmul(box_src.rotation, v) + box_src.translation
                self._intersection_points.append(point_w)

    def intersect_box_poly(self, box, poly):
        """Clips the polygon against the faces of the axis-aligned box."""
        for axis in range(3):
            poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
            poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
        return poly

    def _clip_poly(self, poly, plane, normal, axis):
        """
        Clips the polygon with the plane using the Sutherland-Hodgman algorithm.
        See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of the Sutherland-Hodgman algorithm.
        Here we adopted a robust implementation from "Real-Time Collision Detection", by Christer Ericson, page 370.
        Args:
            poly: List of 3D vertices defining the polygon.
            plane: The 3D vertices of the (2D) axis-aligned plane.
            normal: normal
            axis: A tuple defining a 2D axis.
        Returns:
            List of 3D vertices of the clipped polygon.
        """
        # The vertices of the clipped polygon are stored in the result list.
        result = []
        if len(poly) <= 1:
            return result

        # polygon is fully located on clipping plane
        poly_in_plane = True

        # Test all the edges in the polygon against the clipping plane.
        for i, current_poly_point in enumerate(poly):
            prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
            d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
            d2 = self._classify_point_to_plane(current_poly_point, plane, normal, axis)
        
            if d2 == _POINT_BEHIND_PLANE:
                poly_in_plane = False
                if d1 == _POINT_IN_FRONT_OF_PLANE:
                    intersection = self._intersect(plane, prev_poly_point, current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
            
            elif d2 == _POINT_IN_FRONT_OF_PLANE:
                poly_in_plane = False
                if d1 == _POINT_BEHIND_PLANE:
                    intersection = self._intersect(plane, prev_poly_point, current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
                result.append(current_poly_point)
            
            else:
                if d1 != _POINT_ON_PLANE:
                    result.append(current_poly_point)

        if poly_in_plane:
            return poly
        else:
            return result

    def _intersect(self, plane, prev_point, current_point, axis):
        """
        Computes the intersection of a line with an axis-aligned plane.
        Args:
            plane: Formulated as two 3D points on the plane.
            prev_point: The point on the edge of the line.
            current_point: The other end of the line.
            axis: A tuple defining a 2D axis.
        Returns:
            A 3D point intersection of the poly edge with the plane.
        """
        alpha = (current_point[axis] - plane[axis]) / (current_point[axis] - prev_point[axis])
        # Compute the intersecting points using linear interpolation (lerp)
        intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
        return intersection_point

    def _inside(self, plane, point, axis):
        """Check whether a given point is on a 2D plane."""
        # Cross products to determine the side of the plane the point lie.
        x, y = axis
        u = plane[0] - point
        v = plane[1] - point

        a = u[x] * v[y]
        b = u[y] * v[x]
        return a >= b

    def _classify_point_to_plane(self, point, plane, normal, axis):
        """
        Classify position of a point w.r.t the given plane.
        See Real-Time Collision Detection, by Christer Ericson, page 364.
        Args:
            point: 3x1 vector indicating the point
            plane: 3x1 vector indicating a point on the plane
            normal: scalar (+1, or -1) indicating the normal to the vector
            axis: scalar (0, 1, or 2) indicating the xyz axis
        Returns:
            Side: which side of the plane the point is located.
        """
        signed_distance = normal * (point[axis] - plane[axis])
        if signed_distance > _PLANE_THICKNESS_EPSILON:
            return _POINT_IN_FRONT_OF_PLANE
        elif signed_distance < -_PLANE_THICKNESS_EPSILON:
            return _POINT_BEHIND_PLANE
        else:
            return _POINT_ON_PLANE
