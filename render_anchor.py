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
import os
import torch
from os import makedirs
import numpy as np

from scene import AnchorScene
import time
from gaussian_renderer import anchor_render, anchor_prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.pose_utils import generate_ellipse_path, pose_spherical
from utils.graphics_utils import getWorld2View2
from argparse import ArgumentParser
from arguments import AnchorModelParams, PipelineParams, get_combined_args
from gaussian_renderer import AnchorGaussianModel
import imageio


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        voxel_visible_mask = anchor_prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = anchor_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        depth = render_pkg["depth"]
        depth = depth / (depth.max() + 1e-5)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize();
        t0 = time.time()
        voxel_visible_mask = anchor_prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = anchor_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();
        t1 = time.time()

        t_list.append(t1 - t0)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')


def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    view = views[0]
    renderings = []
    for idx, pose in enumerate(tqdm(generate_ellipse_path(views, n_frames=600), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(
            getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        voxel_visible_mask = anchor_prefilter_voxel(view, gaussians, pipeline, background)
        rendering = anchor_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_all(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, "interpolate_all_{}".format(iteration), "depth")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    frame = 520
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4) for angle in np.linspace(-180, 180, frame + 1)[:-1]], 0)
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

        voxel_visible_mask = anchor_prefilter_voxel(view, gaussians, pipeline, background)
        rendering = anchor_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        # depth = results["depth"]
        # depth = depth / (depth.max() + 1e-5)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset: AnchorModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool,
                skip_test: bool, mode: str):
    with torch.no_grad():
        gaussians = AnchorGaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                        dataset.update_init_factor, dataset.update_hierachy_factor)
        scene = AnchorScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "real-360":
            render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                         background)
        elif mode == "syn-360":
            interpolate_all(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                            background)
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                           background)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                           background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = AnchorModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'syn-360', 'real-360'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
