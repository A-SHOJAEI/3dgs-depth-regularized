"""COLMAP data loader for 3D Gaussian Splatting training."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .colmap_reader import qvec2rotmat, read_colmap_scene
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP scene parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

        cameras, images, points3D = read_colmap_scene(colmap_dir)

        # Build camera intrinsics and extrinsics
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        imsize_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for k in images:
            im = images[k]
            rot = qvec2rotmat(im.qvec)
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = cameras[camera_id]
            if cam.model in ("SIMPLE_PINHOLE",):
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif cam.model in ("PINHOLE",):
                fx, fy = cam.params[0], cam.params[1]
                cx, cy = cam.params[2], cam.params[3]
            elif cam.model in ("SIMPLE_RADIAL",):
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif cam.model in ("RADIAL",):
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif cam.model in ("OPENCV",):
                fx, fy = cam.params[0], cam.params[1]
                cx, cy = cam.params[2], cam.params[3]
            else:
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(f"[Parser] {len(images)} images, {len(set(camera_ids))} cameras.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by image names
        image_names = [images[k].name for k in images]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load bounds if available
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Image paths
        if factor > 1:
            image_dir = os.path.join(data_dir, f"images_{factor}")
        else:
            image_dir = os.path.join(data_dir, "images")

        if not os.path.exists(image_dir):
            # Fallback: use original images
            image_dir = os.path.join(data_dir, "images")

        image_files = sorted(_get_rel_paths(image_dir))
        colmap_image_dir = os.path.join(data_dir, "images")
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))

        image_paths = []
        for f in image_names:
            if f in colmap_to_image:
                image_paths.append(os.path.join(image_dir, colmap_to_image[f]))
            else:
                image_paths.append(os.path.join(image_dir, f))

        # 3D points
        pts_xyz = np.array([points3D[k].xyz for k in points3D]).astype(np.float32)
        pts_rgb = np.array([points3D[k].rgb for k in points3D]).astype(np.uint8)
        pts_err = np.array([points3D[k].error for k in points3D]).astype(np.float32)

        # Normalize world space
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            pts_xyz = transform_points(T1, pts_xyz)

            T2 = align_principal_axes(pts_xyz)
            camtoworlds = transform_cameras(T2, camtoworlds)
            pts_xyz = transform_points(T2, pts_xyz)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.imsize_dict = imsize_dict
        self.points = pts_xyz
        self.points_err = pts_err
        self.points_rgb = pts_rgb
        self.transform = transform

        # Adjust intrinsics if actual image size differs from COLMAP
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # Scene scale
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Camera indices
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_indices = [unique_camera_ids.index(cid) for cid in camera_ids]
        self.exposure_values = [None] * len(image_paths)


class Dataset:
    """Simple dataset class for training/evaluation."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworlds = self.parser.camtoworlds[index]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }

        return data
