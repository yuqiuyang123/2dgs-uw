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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# === 引入工具 ===
import torchvision
import open3d as o3d
from utils.mesh_utils import GaussianExtractor, post_process_mesh

# === [新增] 引入绘图库用于深度图上色 ===
import numpy as np
import matplotlib.pyplot as plt

# === 引入 SeaSplat ===
from seasplat_utils.models import BackscatterNetV2, AttenuateNetV3
from seasplat_utils.losses import BackscatterLoss, GrayWorldLoss, SaturationLoss, AlphaBackgroundLoss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# === 自动渲染函数 ===
def render_seasplat_set(model_path, name, iteration, views, gaussians, pipe, background, bs_model, at_model):
    base_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    
    dry_dir = os.path.join(base_dir, "restored_dry")
    wet_dir = os.path.join(base_dir, "synthesized_wet")
    gt_dir = os.path.join(base_dir, "gt")
    att_dir = os.path.join(base_dir, "attenuation_map")
    bs_dir = os.path.join(base_dir, "backscatter_map")
    depth_dir = os.path.join(base_dir, "depth")

    os.makedirs(dry_dir, exist_ok=True)
    os.makedirs(wet_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(att_dir, exist_ok=True)
    os.makedirs(bs_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"[自动渲染] 正在渲染 {name} 集...")
    
    for idx, view in enumerate(tqdm(views, desc=f"渲染 {name}")):
        render_pkg = render(view, gaussians, pipe, background)
        
        image_dry = render_pkg["render"]
        depth_map = render_pkg["surf_depth"]
        gt_image = view.original_image[0:3, :, :]

        torchvision.utils.save_image(image_dry, os.path.join(dry_dir, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, '{0:05d}.png'.format(idx)))
        
        # ================= [修改] 深度图保存 (Viridis 色调) =================
        # 1. 归一化深度图到 0-1
        depth_vis = depth_map.detach().clone().squeeze() # [H, W]
        depth_max = depth_vis.max()
        if depth_max > 0:
            depth_vis = depth_vis / depth_max
        
        # 2. 转换为 Numpy
        depth_np = depth_vis.cpu().numpy()
        
        # 3. 应用 Viridis Colormap (返回 RGBA, 取前3通道 RGB)
        depth_colored_np = plt.get_cmap('viridis')(depth_np)[:, :, :3]
        
        # 4. 转回 Tensor 并调整维度 [H, W, 3] -> [3, H, W]
        depth_colored = torch.from_numpy(depth_colored_np).permute(2, 0, 1)
        
        # 5. 保存彩色深度图
        torchvision.utils.save_image(depth_colored, os.path.join(depth_dir, '{0:05d}.png'.format(idx)))
        # =================================================================

        if bs_model is not None and at_model is not None:
            # 动态归一化深度用于物理计算
            depth_input = depth_map.unsqueeze(0) / (depth_map.max() + 1e-6)
            rgb_batch = image_dry.unsqueeze(0)

            attenuation = at_model(depth_input)
            backscatter = bs_model(depth_input)

            direct_signal = rgb_batch * attenuation
            image_wet = direct_signal + backscatter
            image_wet = torch.clamp(image_wet, 0.0, 1.0)

            torchvision.utils.save_image(image_wet.squeeze(0), os.path.join(wet_dir, '{0:05d}.png'.format(idx)))
            
            att_vis = attenuation.squeeze(0)
            att_vis = att_vis / (att_vis.max() + 1e-6)
            torchvision.utils.save_image(att_vis, os.path.join(att_dir, '{0:05d}.png'.format(idx)))
            
            bs_vis = backscatter.squeeze(0)
            bs_vis = bs_vis / (bs_vis.max() + 1e-6)
            torchvision.utils.save_image(bs_vis, os.path.join(bs_dir, '{0:05d}.png'.format(idx)))

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bs_model = None
    at_model = None
    physics_optimizer = None
    
    if opt.use_seasplat:
        bs_model = BackscatterNetV2().cuda()
        at_model = AttenuateNetV3().cuda()
        physics_optimizer = torch.optim.Adam([
            {'params': bs_model.parameters()},
            {'params': at_model.parameters()}
        ], lr=opt.seasplat_lr)
        
        loss_fn_bs = BackscatterLoss()
        loss_fn_gw = GrayWorldLoss()
        loss_fn_sat = SaturationLoss()
        loss_fn_alpha = AlphaBackgroundLoss()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 1. 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image_dry = render_pkg["render"]
        depth_map = render_pkg["surf_depth"]
        gt_image = viewpoint_cam.original_image.cuda()
        
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        loss_bs = torch.tensor(0.0).cuda()

        # 2. SeaSplat 物理过程
        if opt.use_seasplat and iteration > opt.start_seasplat_iter:
            # [关键] 动态深度缩放 (0-1)
            depth_max = depth_map.max().detach()
            depth_input = depth_map.unsqueeze(0) / (depth_max + 1e-6)
            rgb_batch = image_dry.unsqueeze(0)
            
            attenuation = at_model(depth_input)
            backscatter = bs_model(depth_input)
            
            direct_signal = rgb_batch * attenuation
            image_wet = direct_signal + backscatter
            image_wet = torch.clamp(image_wet, 0.0, 1.0).squeeze(0)
            
            # Reconstruction Loss
            Ll1 = l1_loss(image_wet, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_wet, gt_image))
            
            # Physics Losses
            bs_detached = bs_model(depth_input.detach())
            direct_detached = gt_image.unsqueeze(0) - bs_detached
            loss_bs = loss_fn_bs(direct_detached)
            loss_gw = loss_fn_gw(rgb_batch)
            loss_sat = loss_fn_sat(rgb_batch)
            
            # Alpha Background Loss
            b_inf_val = torch.sigmoid(bs_model.B_inf).detach()
            loss_alpha = loss_fn_alpha(image_wet, b_inf_val, render_pkg["rend_alpha"])
            
            total_loss = loss + opt.lambda_bs * loss_bs + opt.lambda_gw * loss_gw + 0.1 * loss_sat + 0.01 * loss_alpha
        else:
            # 干模式 / 预热
            Ll1 = l1_loss(image_dry, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_dry, gt_image))
            total_loss = loss

        # 3. 几何正则化 (Normal & Distortion)
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        total_loss += dist_loss + normal_loss
        
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                progress_bar.update(10)
            
            # === [新增] Physics Monitor: 监控物理参数是否在动 ===
            if iteration % 100 == 0:
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    att_mean = torch.sigmoid(at_model.attenuation_conv_params).mean().item()
                    bs_mean = torch.sigmoid(bs_model.backscatter_conv_params).mean().item()
                    # 直接打印到控制台
                    tqdm.write(f"\n[物理监控] 迭代 {iteration}: 衰减参数(Sigmoid)={att_mean:.4f}, 散射参数(Sigmoid)={bs_mean:.4f}")
            # =================================================

            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    tb_writer.add_scalar('train_loss_patches/bs_loss', loss_bs.item(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[迭代 {}] 保存高斯模型".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # [新增创新点代码] 深度感知梯度增强 (Depth-Aware Gradient Boosting)
                # =========================================================
                if opt.use_seasplat and viewspace_point_tensor.grad is not None:
                    # 1. 获取所有 2D 高斯圆盘在 3D 世界中的真实坐标
                    # gaussians.get_xyz: [N, 3]
                    cam_center = viewpoint_cam.camera_center.cuda()
                    
                    # 2. 计算每个圆盘到相机的欧氏距离 (深度)
                    dists = torch.norm(gaussians.get_xyz - cam_center, dim=1)
                    
                    # 3. 计算自适应权重
                    # 逻辑：距离 5米以内权重为1(不放大)，超过5米开始线性放大，最大放大5倍
                    # 你可以根据场景大小调整分母 "5.0"
                    dist_factor = dists / 5.0  
                    grad_weight = torch.clamp(dist_factor, min=1.0, max=5.0)
                    
                    # 4. 原地修改梯度 (In-place modification)
                    # 只有在视野内的点梯度才有意义，利用 visibility_filter 进行掩码操作更安全
                    # viewspace_point_tensor.grad: [N, 2]
                    viewspace_point_tensor.grad[visibility_filter] *= grad_weight[visibility_filter].unsqueeze(1)
                # =========================================================
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    physics_optimizer.step()
                    physics_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[迭代 {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if opt.use_seasplat:
                    torch.save(bs_model.state_dict(), scene.model_path + "/bs_model_" + str(iteration) + ".pth")
                    torch.save(at_model.state_dict(), scene.model_path + "/at_model_" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {"#": gaussians.get_opacity.shape[0], "loss": ema_loss_for_log}
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

    # --------------------------------------------------------------------------------
    # 自动渲染流程
    # --------------------------------------------------------------------------------
    print("\n[训练完成] 开始自动渲染...")
    
    if opt.use_seasplat:
        render_seasplat_set(dataset.model_path, "train", opt.iterations, scene.getTrainCameras(), gaussians, pipe, background, bs_model, at_model)
        if len(scene.getTestCameras()) > 0:
             render_seasplat_set(dataset.model_path, "test", opt.iterations, scene.getTestCameras(), gaussians, pipe, background, bs_model, at_model)
    
    # 自动提取网格
    print("\n[自动渲染] 开始提取网格(Mesh)...")
    mesh_out_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(opt.iterations))
    os.makedirs(mesh_out_dir, exist_ok=True)
    
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    gaussExtractor.reconstruction(scene.getTrainCameras()) 
    
    depth_trunc = gaussExtractor.radius * 2.0
    voxel_size = depth_trunc / 1024
    sdf_trunc = 5.0 * voxel_size
    
    print(f"网格参数: 体素大小(voxel_size)={voxel_size:.4f}, 深度截断(depth_trunc)={depth_trunc:.2f}")
    
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    
    # [修改] 空网格检查，防止报错
    if len(mesh.vertices) > 0:
        o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, 'fuse.ply'), mesh)
        try:
            mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
            o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, 'fuse_post.ply'), mesh_post)
            print("后处理后的网格已保存。")
        except Exception as e:
            print(f"[警告] 网格后处理失败: {e}")
    else:
        print("[警告] 提取的网格为空（0顶点）。跳过后处理。")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print("输出文件夹: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard 不可用: 不记录进度")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        depth = depth / depth.max()
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[迭代 {}] 评估 {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # [关键] 自动对齐 checkpoint
    args.save_iterations.extend(args.checkpoint_iterations) 
    
    print("正在优化 " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\n训练完成。")