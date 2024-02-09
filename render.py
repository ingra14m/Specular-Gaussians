#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, SpecularModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time


def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, specular, use_filter):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    t_list = []
    voxel_visible_mask = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if use_filter:
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
        if use_filter:
            mlp_color = specular.step(gaussians.get_asg_features[voxel_visible_mask], dir_pp_normalized[voxel_visible_mask], normal[voxel_visible_mask])
        else:
            mlp_color = specular.step(gaussians.get_asg_features, dir_pp_normalized, normal)
        results = render(view, gaussians, pipeline, background, mlp_color, voxel_visible_mask=voxel_visible_mask)
        normal_image = render(view, gaussians, pipeline, background, normal[voxel_visible_mask] * 0.5 + 0.5, hybrid=False, voxel_visible_mask=voxel_visible_mask)["render"]
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_image, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()

        if use_filter:
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
        if use_filter:
            mlp_color = specular.step(gaussians.get_asg_features[voxel_visible_mask], dir_pp_normalized[voxel_visible_mask], normal[voxel_visible_mask])
        else:
            mlp_color = specular.step(gaussians.get_asg_features, dir_pp_normalized, normal)
        results = render(view, gaussians, pipeline, background, mlp_color, voxel_visible_mask=voxel_visible_mask)
        
        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')


def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, specular, use_filter):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 520
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background) if use_filter else torch.ones_like(gaussians.get_xyz)[..., 0].bool()
        dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        normal, _ = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
        mlp_color = specular.step(gaussians.get_asg_features[voxel_visible_mask], dir_pp_normalized[voxel_visible_mask],
                                  normal[voxel_visible_mask])
        results = render(view, gaussians, pipeline, background, mlp_color, voxel_visible_mask=voxel_visible_mask)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, opt: OptimizationParams, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        specular = SpecularModel()
        specular.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "all":
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, specular, opt.use_filter)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, specular, opt.use_filter)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, op.extract(args), pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
