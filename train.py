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
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui, prefilter_voxel
import sys
from scene import Scene, GaussianModel, SpecularModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import get_linear_noise_func
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_sets
from metrics import evaluate
import lpips

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
    specular_mlp.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    last_ssim = 0
    last_lpips = 0
    use_filter = dataset.is_real
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    voxel_visible_mask = None
    for iteration in range(1, opt.iterations + 1):

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        N = gaussians.get_xyz.shape[0]

        if use_filter:
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

        if iteration > 3000:
            dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
            if use_filter:
                mlp_color = specular_mlp.step(gaussians.get_asg_features[voxel_visible_mask],
                                              dir_pp_normalized[voxel_visible_mask],
                                              normal.detach()[voxel_visible_mask])
            else:
                mlp_color = specular_mlp.step(gaussians.get_asg_features, dir_pp_normalized, normal.detach())
        else:
            mlp_color = 0

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color,
                            voxel_visible_mask=voxel_visible_mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if use_filter:
                gaussians.max_radii2D[voxel_visible_mask] = torch.max(
                    gaussians.max_radii2D[voxel_visible_mask],
                    radii[visibility_filter])
            else:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), specular_mlp,
                                       dataset.load2gpu_on_the_fly, use_filter)

            if iteration in testing_iterations:
                if iteration == testing_iterations[-1]:
                    cur_psnr, last_ssim, last_lpips = test_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                                                  iter_start.elapsed_time(iter_end),
                                                                  testing_iterations, scene, render, (pipe, background),
                                                                  specular_mlp,
                                                                  dataset.load2gpu_on_the_fly, use_filter)
                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                specular_mlp.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter, voxel_visible_mask,
                                                  use_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                specular_mlp.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                specular_mlp.optimizer.zero_grad()
                specular_mlp.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}, SSIM = {}, LPIPS = {}".format(best_psnr, best_iteration, last_ssim,
                                                                         last_lpips))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def test_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                renderArgs, specular_mlp, load2gpu_on_the_fly, use_filter):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    voxel_visible_mask = None
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = {'name': 'test', 'cameras': scene.getTestCameras()}

        if config['cameras'] and len(config['cameras']) > 0:
            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            for idx, viewpoint in enumerate(config['cameras']):
                if load2gpu_on_the_fly:
                    viewpoint.load2device()

                if use_filter:
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                dir_pp = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(
                    scene.gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = scene.gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
                if use_filter:
                    mlp_color = specular_mlp.step(scene.gaussians.get_asg_features[voxel_visible_mask],
                                                  dir_pp_normalized[voxel_visible_mask], normal[voxel_visible_mask])
                else:
                    mlp_color = specular_mlp.step(scene.gaussians.get_asg_features, dir_pp_normalized, normal)

                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs, mlp_color,
                               voxel_visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                lpips_test += lpips_fn(image, gt_image).mean().double()

                if load2gpu_on_the_fly:
                    viewpoint.load2device('cpu')
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                         image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                             gt_image[None], global_step=iteration)

            l1_test /= len(config['cameras'])
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])

            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return psnr_test, ssim_test, lpips_test


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, specular_mlp, load2gpu_on_the_fly, use_filter):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    voxel_visible_mask = None
    # Report test and samples of training set
    if iteration in testing_iterations[:-1]:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()

                    if use_filter:
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    dir_pp = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(
                        scene.gaussians.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    normal = scene.gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
                    if use_filter:
                        mlp_color = specular_mlp.step(scene.gaussians.get_asg_features[voxel_visible_mask],
                                                      dir_pp_normalized[voxel_visible_mask], normal[voxel_visible_mask])
                    else:
                        mlp_color = specular_mlp.step(scene.gaussians.get_asg_features, dir_pp_normalized, normal)

                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, mlp_color,
                                   voxel_visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000] + list(range(20000, 30001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")

    # rendering
    print(f'\nStarting Rendering~')
    render_sets(lp.extract(args), -1, op.extract(args), pp.extract(args), skip_train=True, skip_test=False,
                mode="render")
    print("\nRendering complete.")

    # calc metrics
    # print("\nStarting evaluation...")
    # evaluate([str(args.model_path)])
    # print("\nEvaluating complete.")
