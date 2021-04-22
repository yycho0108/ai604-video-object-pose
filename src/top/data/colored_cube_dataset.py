#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK

import itertools
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Union, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from pytorch3d.structures import Pointclouds
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from top.run.torch_util import resolve_device
from top.run.app_util import update_settings


class ColoredCubeDataset(th.utils.data.Dataset):
    """
    Toy generative dataset for 3D object detection:
    vertices of an oriented unit cube rendered as a point cloud.
    Intended to be an "easy" baseline.

    # FIXME(ycho): I realized that the cube length is actually 2
    """

    @dataclass
    class Settings(Serializable):
        batch_size: int = 1
        aspect: float = 1.0  # pixel aspect ratio, max_x/max_y
        fov: float = 60  # full vertical field of view, in degrees.
        znear: float = 0.1
        zfar: float = 100.0
        min_distance: float = 0.1
        max_distance: float = 10.0
        image_size: Tuple[int, int] = (256, 256)  # Order: H W

    def __init__(self, opts: Settings, device: th.device = '', transform=None):
        super().__init__()
        self.opts = opts
        self.device = resolve_device(device)
        self.xfm = transform
        # TODO(ycho): Consider support for multiple "objects".
        # TODO(ycho): Consider support for *animated* objects through time.
        self.cloud = self._get_point_cloud()
        self.clouds = self.cloud.extend(self.opts.batch_size)
        self.renderer = self._setup_render()
        self.tan_half_fov = np.tan(np.deg2rad(0.5 * self.opts.fov))

        self.min_distance = np.maximum(
            opts.min_distance,
            (0.5 * np.sqrt(3)) / self.tan_half_fov)

    def _setup_render(self):
        # Unpack options ...
        opts = self.opts

        # Initialize a camera.
        # TODO(ycho): Alternatively, specify the intrinsic matrix `K` instead.
        cameras = FoVPerspectiveCameras(
            znear=opts.znear,
            zfar=opts.zfar,
            aspect_ratio=opts.aspect,
            fov=opts.fov,
            degrees=True,
            device=self.device
        )

        # Define the settings for rasterization and shading.
        # As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to raster_points.py for explanations of
        # these parameters.
        # points_per_pixel (Optional): We will keep track of this many points per
        # pixel, returning the nearest points_per_pixel points along the z-axis
        raster_settings = PointsRasterizationSettings(
            image_size=opts.image_size,
            radius=0.1,
            points_per_pixel=8
        )

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        return renderer

    def _get_point_cloud(self):
        """
        Get vertices of a unit-cube, with colors assigned according to vertex coordinates.
        """
        vertices = list(itertools.product(
            *zip([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])))
        vertices = th.as_tensor(vertices, dtype=th.float32, device=self.device)
        vertices = vertices

        # Map vertices to colors. =RGB(0.25~0.75)
        colors = (0.5 + 0.5 * vertices)
        cloud = Pointclouds(points=vertices[None], features=colors[None])
        return cloud

    def _render(self):
        """
        Render the unit cube at various camera poses.
        We deliberately sample the camera poses such that all vertices of the cube
        are always in view.
        """
        opts = self.opts

        # TODO(ycho): is `no_grad()` here necessary?
        with th.no_grad():
            # Sample a Unit-ray viewing direction.
            ray = th.randn(size=(opts.batch_size, 3), device=self.device)
            ray /= th.norm(ray, dim=1, keepdim=True)

            # Compute distance along the ray according to constraints.
            distance = (self.min_distance +
                        (opts.max_distance - self.min_distance) *
                        th.rand(size=(opts.batch_size, 1), device=self.device))

            # NOTE(ycho): `sqrt(3)/2` here comes from max radius of unit cube.
            # The generic thing to do would be to compute the radius of our
            # cloud. We're not explicitly taking the cube into account.
            max_tangential_offset = th.clamp(
                distance *
                self.tan_half_fov -
                (0.5 * np.sqrt(3)),
                min=0.0)

            # NOTE(ycho): We're sampling an offset orthogonal to the view ray.
            # The constraint here is to be visible within the view plane.
            offset = th.randn(size=(opts.batch_size, 3), device=self.device)
            offset = offset - th.einsum('...a,...a->...',
                                        offset, ray)[:, None] * ray
            offset *= (max_tangential_offset /
                       th.norm(offset, dim=1, keepdim=True))

            # Finally, we have the full definition of the ray.
            at = offset
            pos = offset + ray * distance

            # Compose the `lookat` camera transform according to this position
            R, T = look_at_view_transform(
                eye=pos, at=at, device=self.device)
            img = self.renderer(
                point_clouds=self.clouds, R=R, T=T
            )

            # pytorch3d uses `NHWC` convension, convert to NCHW.
            img = img.permute(0, 3, 1, 2)

            # pytorch3d uses a `float` tensor for representing images,
            # whereas we'd like for the image to come out as `uint8` ONLY for
            # compatibility with the output from the `Objectron` dataset.
            img = img.mul_(255.0).to(dtype=th.uint8)

        return (img, R, T)

    def __iter__(self):
        while True:
            (img, R, T) = self._render()
            points_2d = self.renderer.rasterizer.transform(
                self.clouds, R=R, T=T)

            # Convert `points` to UV coordinates.
            # This is to be consistent with the objectron dataset convention.
            points_2d = points_2d.points_padded()
            points_2d[..., :2] *= -0.5
            points_2d[..., :2] += 0.5

            # Add an axis indicating num_instance == 0
            points_2d = points_2d[:, None]

            # Projection matrix maps to -1 ~ 1 NDC coordinates.
            P = self.renderer.rasterizer.cameras.get_projection_transform()
            P = P.get_matrix()

            # NOTE(ycho): Permute projection matrix to fit objectron
            # convention.
            # TODO(ycho): Verify if this keeps +z-axis sign convention
            # consistent.
            permutation = th.as_tensor([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]], dtype=th.float32, device=self.device)
            projection = permutation @ P.transpose(2, 1)

            # NOTE(ycho): Swap conventions here...
            # rotation matrix : rmul -> lmul
            # NOTE(ycho): Also, return flattened output as in `Objectron`.
            orientation = (R.transpose(2, 1).reshape(self.opts.batch_size, -1))

            # TODO(ycho): Figure out a way to unify these formats.
            # see ai604-video-object-pose#10
            out = {
                'image': img,
                'object/orientation': orientation,
                'object/translation': T,
                'object/scale': th.full_like(T, 1.0),
                'points': points_2d,
                'num_instances': th.ones(
                    self.opts.batch_size,
                    device=self.device),
                'camera/projection': projection,
            }
            if self.xfm is not None:
                out = self.xfm(out)
            yield out


def main():
    opts = ColoredCubeDataset.Settings(batch_size=4)
    opts = update_settings(opts)

    device = resolve_device()
    dataset = ColoredCubeDataset(opts, device,)
    for data in dataset:
        save_image(data['image'] / 255.0, F'/tmp/img.png')
        break


if __name__ == '__main__':
    main()
