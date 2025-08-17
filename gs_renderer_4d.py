# Copyright (c) 2024 GenMOJO and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import scipy
from scipy.spatial.transform import Rotation as R

from diff_gauss import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from utils.sh_utils import eval_sh, SH2RGB

from gaussian_model_4d import GaussianModel, BasicPointCloud
from arguments import ModelHiddenParams
from utils.general_utils import get_expon_lr_func, build_rotation, quat_mult, point_cloud_to_image
from scene.deformation_sh import initialize_zeros_weights

class Renderer:
    def __init__(self, T, sh_degree=3, white_background=True, radius=1):

        assert isinstance(T, int)
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.T = T

        self.gaussians = dict()
        self.gaussian_bounds = dict()

        self.gaussian_depth_scaling = dict() 
        self.gaussian_translations = dict()
        self.gaussian_scales = dict()

        self.nn_weights = dict()
        self.nn_mask = dict()
        self.nn_indices = dict()

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def add_object(self, ply_path, name, init_zero_deformation=False):

        hyper = ModelHiddenParams(None) # args
        # Overwrite K-plane time resolution based on video length
        if int(0.8*self.T) > 25:
            hyper.kplanes_config['resolution'] = hyper.kplanes_config['resolution'][:-1] +  [int(0.8*self.T)]
        self.gaussians[name] = GaussianModel(self.sh_degree, hyper)
        self.gaussians[name].load_ply(ply_path)

        self.precompute_nn(name, k=16, dist_thresh=0.05)

        means3D = self.gaussians[name].get_xyz
        lower_bounds = torch.quantile(means3D, q=0.1, dim=0)
        center = torch.quantile(means3D, q=0.5, dim=0)
        upper_bounds = torch.quantile(means3D, q=0.9, dim=0)
        self.gaussian_bounds[name] = {
            'lower': lower_bounds,
            'center': center,
            'upper': upper_bounds
        }
        # set deformation to zero
        if init_zero_deformation:
            self.gaussians[name]._deformation.apply(initialize_zeros_weights)

    def precompute_nn(self, obj_name, k=16, dist_thresh=0.05):
        means3D = self.gaussians[obj_name].get_xyz
        opacity_mask = (self.gaussians[obj_name].get_opacity > 0.).squeeze(1)
        vis_means3D = means3D[opacity_mask].detach().cpu().numpy()
        means_kdtree = scipy.spatial.cKDTree(vis_means3D, leafsize=100)
        nearest_neighbors = means_kdtree.query(vis_means3D, k=k)
        self.nn_mask[obj_name] = (nearest_neighbors[0] < dist_thresh)
        self.nn_indices[obj_name] = nearest_neighbors[1]
        self.nn_weights[obj_name] = torch.from_numpy(np.exp(-10. * nearest_neighbors[0])).to(means3D.device)

    def init_obj_warps(self, translation, scale, depth_scale, name):

        self.gaussian_translations[name] = translation
        self.gaussian_scales[name] = scale
        self.gaussian_depth_scaling[name] = depth_scale

        self.gaussian_translations[name].requires_grad_(True)
        self.gaussian_scales[name].requires_grad_(False)
        self.gaussian_depth_scaling[name].requires_grad_(True)
        
    def create_joint_optimizer(self, opt, lr_scale, specify_obj=None):

        if isinstance(specify_obj, str):
            specify_obj = [specify_obj]
        
        for i, obj_name in enumerate(self.gaussians.keys()):

            if specify_obj is not None and obj_name not in specify_obj:
                continue

            self.gaussians[obj_name].spatial_lr_scale = 1.0 #1.0 
            self.gaussians[obj_name].training_setup(opt)

            if i == 0:
                # Setup optimizer
                l = [
                    {'params': [self.gaussians[obj_name]._features_dc], 'lr': opt.feature_lr / 10.0, "name": f"{obj_name}_f_dc"},
                    {'params': [self.gaussians[obj_name]._features_rest], 'lr': opt.feature_lr / 20.0, "name": f"{obj_name}_f_rest"},
                    {'params': list(self.gaussians[obj_name]._deformation.get_mlp_parameters()), 'lr': opt.deformation_lr_init * self.gaussians[obj_name].spatial_lr_scale, "name": f"{obj_name}_deformation"},
                    {'params': list(self.gaussians[obj_name]._deformation.get_grid_parameters()), 'lr': opt.grid_lr_init * self.gaussians[obj_name].spatial_lr_scale, "name": f"{obj_name}_grid"},
                    {'params': [self.gaussian_translations[obj_name]], 'lr': opt.deformation_lr_init * lr_scale, 'weight_decay': 0, "name": f"{obj_name}_translation"},
                    {'params': [self.gaussian_depth_scaling[obj_name]], 'lr': opt.deformation_lr_init * lr_scale, 'weight_decay': 0, "name": f"{obj_name}_depth_scaling"},
                ]
                self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
            else:
                # Add params
                self.optimizer.add_param_group({
                    'params': [self.gaussians[obj_name]._features_dc],
                    'lr': opt.feature_lr * lr_scale,
                    "name": f"{obj_name}_f_dc"
                })
                self.optimizer.add_param_group({
                    'params': [self.gaussians[obj_name]._features_rest],
                    'lr': opt.feature_lr * lr_scale / 20.0,
                    "name": f"{obj_name}_f_rest"
                })
                self.optimizer.add_param_group({
                    'params': list(self.gaussians[obj_name]._deformation.get_mlp_parameters()),
                    'lr': opt.deformation_lr_init * self.gaussians[obj_name].spatial_lr_scale,
                    "name": f"{obj_name}_deformation"
                })
                self.optimizer.add_param_group({
                    'params': list(self.gaussians[obj_name]._deformation.get_grid_parameters()),
                    'lr': opt.grid_lr_init * self.gaussians[obj_name].spatial_lr_scale,
                    "name": f"{obj_name}_grid"
                })
                self.optimizer.add_param_group({
                    'params': [self.gaussian_translations[obj_name]],
                    'lr': opt.deformation_lr_init * lr_scale,
                    'weight_decay': 0,
                    "name": f"{obj_name}_translation"
                })
                self.optimizer.add_param_group({
                    'params': [self.gaussian_depth_scaling[obj_name]],
                    'lr': opt.deformation_lr_init * lr_scale,
                    'weight_decay': 0,
                    "name": f"{obj_name}_depth_scaling"
                })

            # Turn on the gradients
            self.gaussians[obj_name].unfreeze()
            self.gaussian_translations[obj_name].requires_grad_(True)
            self.gaussian_depth_scaling[obj_name].requires_grad_(True)

    def prepare_render_all(self, timesteps, specify_obj=None, prev_frame_is_zeroth_frame=False):

        if isinstance(specify_obj, (str, int, np.int64)):
            specify_obj = [specify_obj]
        
        means3D_T_all = {}
        means3D_T_all_prev = {}
        opacity_T_all = {}
        opacity_T_all_prev = {}
        scales_T_all = {}
        scales_T_all_prev = {}
        rotations_T_all = {}
        rotations_T_all_prev = {}

        shs_delta_T_all = {}
        shs_delta_T_all_prev = {}
        
        for obj_name in self.gaussians.keys():

            if specify_obj is not None and obj_name not in specify_obj:
                continue

            means3D = self.gaussians[obj_name].get_xyz
            opacity = self.gaussians[obj_name]._opacity
            scales = self.gaussians[obj_name]._scaling
            rotations = self.gaussians[obj_name]._rotation

            # Aggregrate all inputs into one big input
            means3D_T = []
            opacity_T = []
            scales_T = []
            rotations_T = []
            time_T = []
            prev_time_T = []

            for t in timesteps:
                time = torch.tensor(t).to(means3D.device).repeat(means3D.shape[0], 1)
                time = ((time.float() / self.T) - 0.5) * 2

                if prev_frame_is_zeroth_frame:
                    prev_time = torch.tensor(0).to(means3D.device).repeat(means3D.shape[0], 1)
                else:
                    prev_time = torch.tensor(t-1 if t > 0 else 0).to(means3D.device).repeat(means3D.shape[0], 1)
                prev_time = ((prev_time.float() / self.T) - 0.5) * 2

                time_T.append(time)
                prev_time_T.append(prev_time)

            means3D_T = means3D.repeat(2*len(timesteps), 1)
            scales_T = scales.repeat(2*len(timesteps), 1)
            rotations_T = rotations.repeat(2*len(timesteps), 1)
            opacity_T = opacity.repeat(2*len(timesteps), 1)
            time_T = torch.cat(time_T)
            prev_time_T = torch.cat(prev_time_T)

            # Get the deformations
            (means3D_deform_T,
            scales_deform_T,
            rotations_deform_T,
            opacity_deform_T,
            delta_sh_T) = self.gaussians[obj_name]._deformation(means3D_T, scales_T, 
                                                            rotations_T, opacity_T,
                                                            torch.cat([time_T, prev_time_T], dim=0))
            
            # Split the deformations
            means3D_deform_T, prev_means3D_deform_T = torch.chunk(means3D_deform_T, 2, dim=0)
            opacity_deform_T, prev_opacity_deform_T = torch.chunk(opacity_deform_T, 2, dim=0)
            scales_deform_T, prev_scales_deform_T = torch.chunk(scales_deform_T, 2, dim=0)
            rotations_deform_T, prev_rotations_deform_T = torch.chunk(rotations_deform_T, 2, dim=0)
            delta_sh_T, prev_delta_sh_T = torch.chunk(delta_sh_T, 2, dim=0)

            # Cache the deformations
            num_pts = means3D_deform_T.shape[0] // len(timesteps)
            means3D_deform_T = means3D_deform_T.reshape([len(timesteps), num_pts, -1])
            opacity_deform_T = opacity_deform_T.reshape([len(timesteps), num_pts, -1])
            scales_deform_T = scales_deform_T.reshape([len(timesteps), num_pts, -1])
            rotations_deform_T = rotations_deform_T.reshape([len(timesteps), num_pts, -1])
            delta_sh_T = delta_sh_T.reshape([len(timesteps), num_pts, 1, -1])

            num_pts = prev_means3D_deform_T.shape[0] // len(timesteps)
            prev_means3D_deform_T = prev_means3D_deform_T.reshape([len(timesteps), num_pts, -1])
            prev_opacity_deform_T = prev_opacity_deform_T.reshape([len(timesteps), num_pts, -1])
            prev_scales_deform_T = prev_scales_deform_T.reshape([len(timesteps), num_pts, -1])
            prev_rotations_deform_T = prev_rotations_deform_T.reshape([len(timesteps), num_pts, -1])
            prev_delta_sh_T = prev_delta_sh_T.reshape([len(timesteps), num_pts, 1, -1])
            means3D_deform_T = means3D_deform_T
            prev_means3D_deform_T = prev_means3D_deform_T

            means3D_T_all[obj_name] = means3D_deform_T
            means3D_T_all_prev[obj_name] = prev_means3D_deform_T
            opacity_T_all[obj_name] = opacity_deform_T
            opacity_T_all_prev[obj_name] = prev_opacity_deform_T
            scales_T_all[obj_name] = scales_deform_T
            scales_T_all_prev[obj_name] = prev_scales_deform_T
            rotations_T_all[obj_name] = rotations_deform_T
            rotations_T_all_prev[obj_name] = prev_rotations_deform_T
            shs_delta_T_all[obj_name] = delta_sh_T
            shs_delta_T_all_prev[obj_name] = prev_delta_sh_T

        self.time_deform_T = timesteps
        self.means3D_deform_T = means3D_T_all
        self.opacity_deform_T = opacity_T_all
        self.scales_deform_T = scales_T_all
        self.rotations_deform_T = rotations_T_all
        self.shs_delta_T = shs_delta_T_all

        if prev_frame_is_zeroth_frame:
            self.prev_time_deform_T = [0 for x in timesteps]
        else:
            self.prev_time_deform_T = [(x-1 if x > 0 else 0) for x in timesteps]
        self.prev_means3D_deform_T = means3D_T_all_prev
        self.prev_opacity_deform_T = opacity_T_all_prev
        self.prev_scales_deform_T = scales_T_all_prev
        self.prev_rotations_deform_T = rotations_T_all_prev
        self.prev_shs_delta_T = shs_delta_T_all_prev

    def render_all(
        self,
        viewpoint_camera,
        default_camera_center=None,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        direct_render=True,
        specify_obj=None,
        no_deform=False,
        account_for_global_motion=True,
        virtual2real=None,
        virtual2real_t0 = None,
        cut_gaussians = False
    ): 

        if isinstance(specify_obj, (str, int, np.int64)):
            specify_obj = [specify_obj]
        
        all_gaussians_xyz = []
        for obj_name in self.gaussians.keys():
            if specify_obj is not None and obj_name not in specify_obj:
                continue
            all_gaussians_xyz.append(self.gaussians[obj_name].get_xyz)
        all_gaussians_xyz = torch.cat(all_gaussians_xyz)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                all_gaussians_xyz,
                dtype=self.gaussians[0].get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians[0].active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        all_means3D_final = []
        all_shs = []
        all_colors_precomp = []
        all_opacity = []
        all_scales = []
        all_rotations = []
        all_cov3D_precomp = []
        all_extras = []

        # Process individual object Gaussians
        for obj_id, obj_name in enumerate(self.gaussians.keys()):

            if specify_obj is not None and obj_name not in specify_obj:
                continue

            means3D = self.gaussians[obj_name].get_xyz
            opacity = self.gaussians[obj_name]._opacity
            shs = self.gaussians[obj_name].get_features
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = None
            if compute_cov3D_python:
                cov3D_precomp = self.gaussians[obj_name].get_covariance(scaling_modifier)
            else:
                scales = self.gaussians[obj_name]._scaling
                rotations = self.gaussians[obj_name]._rotation
            
            if no_deform:
                # Query init locations, used for initializing objects
                means3D_final = means3D
                rotations_final = rotations
                scales_final = scales
                shs_final = shs
            else:
                # First get the deformations
                if direct_render:
                    # Directly query network, used for inference
                    (means3D_deform, scales_deform,
                        rotations_deform, opacity_deform, delta_sh) = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                                                rotations, opacity,
                                                                                                time)
                    shs_deform = shs + delta_sh.reshape([shs.shape[0], 1, -1])
                else:
                    # Cached mode, used for training
                    idx = self.time_deform_T.index(viewpoint_camera.time)
                    means3D_deform, scales_deform, rotations_deform = self.means3D_deform_T[obj_name][idx], self.scales_deform_T[obj_name][idx], self.rotations_deform_T[obj_name][idx]
                    delta_sh = self.shs_delta_T[obj_name][idx]
                    shs_deform = shs + delta_sh

                means3D_final = means3D_deform
                rotations_final = rotations_deform
                scales_final = scales_deform
                # opacity_final = opacity_deform
                shs_final = shs_deform

            scales_final = self.gaussians[obj_name].scaling_activation(scales_final)
            rotations_final = self.gaussians[obj_name].rotation_activation(rotations_final)
            opacity = self.gaussians[obj_name].opacity_activation(opacity)
            
            # First go from obj centric frame to world frame
            # Warp Gaussians from obj centric to world frame
            if account_for_global_motion:
                means3D_final, scales_final = self._objcentric2world_warp(
                    means3D_final, scales_final,
                    viewpoint_camera, default_camera_center, obj_name
                )

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            if colors_precomp is None:
                if convert_SHs_python:
                    shs_view = shs_final.transpose(1, 2).view(
                        -1, 3, (self.gaussians[obj_name].max_sh_degree + 1) ** 2
                    )
                    dir_pp = self.gaussians[obj_name].get_xyz - viewpoint_camera.camera_center.repeat(
                        self.gaussians[obj_name].get_features.shape[0], 1
                    )
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(
                        self.gaussians[obj_name].active_sh_degree, shs_view, dir_pp_normalized
                    )
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = shs_final
            else:
                colors_precomp = override_color

            # Filter large Gaussians
            if cut_gaussians and obj_name != 'background':
                scales_mask = torch.max(scales_final, dim=1)[0] <= 0.1
            else:
                # Filter large Gaussians
                scales_mask = torch.max(scales_final, dim=1)[0] <= 1e12

            instance_labels = torch.nn.functional.one_hot(
                torch.tensor([obj_id] * len(means3D_final), device=means3D_final.device),
                num_classes=len(list(self.gaussians.keys()))
            )

            means3D_masked = means3D_final[scales_mask]
            rotations_masked = rotations_final[scales_mask]
            if obj_name == 'background' and virtual2real_t0 is not None:
                means3D_masked, rotations_masked = self._virtual2real_warp(
                    means3D_masked, rotations_masked,
                    virtual2real_t0
                )
            elif virtual2real is not None:
                means3D_masked, rotations_masked = self._virtual2real_warp(
                    means3D_masked, rotations_masked,
                    virtual2real
                )

            all_means3D_final.append(means3D_masked)
            all_shs.append(shs[scales_mask])
            all_colors_precomp.append(colors_precomp)
            all_opacity.append(opacity[scales_mask])
            all_scales.append(scales_final[scales_mask])
            all_rotations.append(rotations_masked)
            all_cov3D_precomp.append(cov3D_precomp)
            all_extras.append(instance_labels[scales_mask])

        means3D_final = torch.cat(all_means3D_final)
        means2D = screenspace_points
        shs = torch.cat(all_shs)
        colors_precomp = None if all_colors_precomp[0] is None else torch.cat(colors_precomp)
        opacity = torch.cat(all_opacity)
        scales_final = None if all_scales[0] is None else torch.cat(all_scales)
        rotations_final = None if all_rotations[0] is None else torch.cat(all_rotations)
        cov3D_precomp = None if all_cov3D_precomp[0] is None else torch.cat(all_cov3D_precomp)
        extra_attrs = torch.cat(all_extras).float()
        
        rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3Ds_precomp = cov3D_precomp,
            extra_attrs = extra_attrs
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "normal": rendered_norm,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "extra": extra,
        }

    def render_flow_all(
        self,
        viewpoint_cameras,
        default_camera_center=None,
        scaling_modifier=1.0,
        bg_color=None,
        virtual2reals=[None, None],
        account_for_global_motion=True,
    ):
        
        assert(len(viewpoint_cameras) == 2)

        # Get Gaussian positions and compute flow
        gaussian_2d_pos_curr, _, gaussian_3d_pos_curr = self.get_2d_gaussian_pos_all(
            viewpoint_cameras[0],
            default_camera_center,
            virtual2reals[0],
            account_for_global_motion
        )
        gaussian_2d_pos_prev, _, gaussian_3d_pos_prev = self.get_2d_gaussian_pos_all(
            viewpoint_cameras[1],
            default_camera_center,
            virtual2reals[1],
            account_for_global_motion
        )
        flow_2d = gaussian_2d_pos_curr - gaussian_2d_pos_prev
        flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, 1:])], dim=1)

        # Collect Gaussians from all objects
        all_gaussians_xyz = []
        all_scales_0 = []
        all_indices = []
        all_weights = []

        for obj_name in self.gaussians.keys():
            all_gaussians_xyz.append(self.gaussians[obj_name].get_xyz)
            all_scales_0.append(self.gaussians[obj_name]._scaling)
            if len(all_indices) > 0:
                all_indices.append(torch.from_numpy(self.nn_indices[obj_name]) + len(torch.cat(all_indices)))
            else:
                all_indices.append(torch.from_numpy(self.nn_indices[obj_name]))

            all_weights.append(self.nn_weights[obj_name])

        all_gaussians_xyz = torch.cat(all_gaussians_xyz)
        all_scales_0 = torch.cat(all_scales_0)
        all_indices = torch.cat(all_indices)
        all_weights = torch.cat(all_weights)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                all_gaussians_xyz,
                dtype=self.gaussians[0].get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_cameras[1].FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cameras[1].FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_cameras[1].image_height),
            image_width=int(viewpoint_cameras[1].image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_cameras[1].world_view_transform,
            projmatrix=viewpoint_cameras[1].full_proj_transform,
            sh_degree=self.gaussians[0].active_sh_degree,
            campos=viewpoint_cameras[1].camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        all_means3D_final = []
        all_opacity = []
        all_scales = []
        all_scales_deform = []
        all_rotations = []
        all_shs_deform = []
        # Process individual object Gaussians
        for obj_name in self.gaussians.keys():

            opacity = self.gaussians[obj_name]._opacity
            # Get Gaussian deformations
            idx = self.prev_time_deform_T.index(viewpoint_cameras[1].time)
            means3D_deform, scales_deform, rotations_deform, opacity_deform = self.prev_means3D_deform_T[obj_name][idx], self.prev_scales_deform_T[obj_name][idx], self.prev_rotations_deform_T[obj_name][idx], self.prev_opacity_deform_T[obj_name][idx]
            all_scales_deform.append(scales_deform) # Need pre-activated scales for loss

            delta_sh_T = self.shs_delta_T[obj_name][idx]
            all_shs_deform.append(delta_sh_T)

            means3D_final = means3D_deform
            rotations_final = rotations_deform
            scales_final = scales_deform
            # opacity_final = opacity_deform

            scales_final = self.gaussians[obj_name].scaling_activation(scales_final)
            rotations_final = self.gaussians[obj_name].rotation_activation(rotations_final)
            opacity = self.gaussians[obj_name].opacity_activation(opacity)

            # Warp Gaussians from obj centric to world frame
            if account_for_global_motion:
                means3D_final, scales_final = self._objcentric2world_warp(
                    means3D_final, scales_final,
                    viewpoint_cameras[1], default_camera_center, obj_name
                )
            # Warp from virtual to real cameras
            if virtual2reals[1] is not None:
                means3D_final, rotations_final = self._virtual2real_warp(
                    means3D_final, rotations_final,
                    virtual2reals[1]
                )
            
            all_means3D_final.append(means3D_final)
            all_opacity.append(opacity)
            all_scales.append(scales_final)
            all_rotations.append(rotations_final)

        means3D_final = torch.cat(all_means3D_final)
        means2D = screenspace_points
        opacity = torch.cat(all_opacity)
        scales_final = torch.cat(all_scales)
        rotations_final = torch.cat(all_rotations)

        # Only update through flow, detach everything else
        rendered_flow_image, _, _, _, _, _ = rasterizer(
            means3D = means3D_final.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = flow_padded,
            opacities = opacity.detach(),
            scales = scales_final.detach(),
            rotations = rotations_final.detach(),
            cov3Ds_precomp = None
        )

        # Compute regularization losses
        shs_change = torch.cat(all_shs_deform)

        scale_change = (torch.cat(all_scales_deform) - all_scales_0)
        local_scale_loss = scale_change[all_indices] - scale_change.unsqueeze(1)
        local_scale_loss = torch.sqrt((local_scale_loss ** 2).sum(-1) * all_weights + 1e-20).mean()
        
        # Get Gaussian rotations for current timestep
        idx = self.time_deform_T.index(viewpoint_cameras[0].time)
        all_curr_rotations_final_inv = []
        for obj_name in self.gaussians.keys():
            curr_rotations_deform = self.rotations_deform_T[obj_name][idx]
            curr_rotations_final = curr_rotations_deform
            curr_rotations_final = self.gaussians[obj_name].rotation_activation(curr_rotations_final)
            curr_rotations_final_inv = curr_rotations_final
            curr_rotations_final_inv[:, 1:] = -1. * curr_rotations_final_inv[:, 1:]
            all_curr_rotations_final_inv.append(curr_rotations_final_inv)
        all_curr_rotations_final_inv = torch.cat(all_curr_rotations_final_inv)
        
        # Local Rigidity Loss from Dynamic Gaussian Splatting
        prev_nn_displacement = gaussian_3d_pos_prev[all_indices] - gaussian_3d_pos_prev.unsqueeze(1)
        curr_nn_displacement = gaussian_3d_pos_curr[all_indices] - gaussian_3d_pos_curr.unsqueeze(1)
        rel_rotmat = build_rotation(quat_mult(rotations_final, all_curr_rotations_final_inv))
        curr_nn_displacement_warped = (rel_rotmat.transpose(2, 1)[:, None] @ curr_nn_displacement[:, :, :, None]).squeeze(-1)
        local_rigidity_loss = torch.sqrt(((prev_nn_displacement - curr_nn_displacement_warped) ** 2).sum(-1) * all_weights + 1e-20).mean()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "flow": rendered_flow_image,
            "viewspace_points": screenspace_points,
            "scale_change": scale_change,
            "sh_change": shs_change,
            "local_scale_loss": local_scale_loss,
            "gaussian_displacements": (gaussian_3d_pos_curr - gaussian_3d_pos_prev),
            'local_rigidity_loss': local_rigidity_loss,
        }
    
    @torch.no_grad()
    def render_3d_pos_all(
        self,
        viewpoint_cameras,
        default_camera_center=None,
        scaling_modifier=1.0,
        bg_color=None,
        compute_cov3D_python=False,
        virtual2real=False,
        account_for_global_motion=True,
    ):
        
        assert(len(viewpoint_cameras) == 2)

        all_gaussians_xyz = []
        for obj_name in self.gaussians.keys():
            all_gaussians_xyz.append(self.gaussians[obj_name].get_xyz)
        all_gaussians_xyz = torch.cat(all_gaussians_xyz)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                all_gaussians_xyz,
                dtype=self.gaussians[0].get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_cameras[1].FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cameras[1].FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_cameras[1].image_height),
            image_width=int(viewpoint_cameras[1].image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_cameras[1].world_view_transform,
            projmatrix=viewpoint_cameras[1].full_proj_transform,
            sh_degree=self.gaussians[0].active_sh_degree,
            campos=viewpoint_cameras[1].camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        all_means3D_final = []
        all_opacity = []
        all_scales = []
        all_rotations = []
        all_new_means3D = []
        # Process individual object Gaussians
        for obj_name in self.gaussians.keys():
            
            # 1. We render on top of prev_cam, so get the information first
            means3D = self.gaussians[obj_name].get_xyz
            prev_time = torch.tensor(viewpoint_cameras[1].time).to(means3D.device).repeat(means3D.shape[0],1)
            prev_time = ((prev_time.float() / self.T) - 0.5) * 2

            opacity = self.gaussians[obj_name]._opacity
            scales = self.gaussians[obj_name]._scaling
            rotations = self.gaussians[obj_name]._rotation
            
            # Directly query network, used for inference
            (means3D_deform, scales_deform,
            rotations_deform, opacity_deform) = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                                    rotations, opacity,
                                                                                    prev_time)

            means3D_final = means3D_deform
            rotations_final = rotations_deform
            scales_final = scales_deform
            # opacity_final = opacity_deform

            scales_final = self.gaussians[obj_name].scaling_activation(scales_final)
            rotations_final = self.gaussians[obj_name].rotation_activation(rotations_final)
            opacity = self.gaussians[obj_name].opacity_activation(opacity)
            
            # Warp Gaussians from obj centric to world frame
            if account_for_global_motion:
                means3D_final, scales_final = self._objcentric2world_warp(
                    means3D_final, scales_final,
                    viewpoint_cameras[1], default_camera_center, obj_name
                )
            # Warp from virtual to real cameras
            if virtual2real is not None:
                means3D_final, rotations_final = self._virtual2real_warp(
                    means3D_final, rotations_final,
                    virtual2real
                )
            
            all_means3D_final.append(means3D_final)
            all_opacity.append(opacity)
            all_scales.append(scales_final)
            all_rotations.append(rotations_final)

            # 2. Now we get the new locations of the points in cur_cam
            cur_time = torch.tensor(viewpoint_cameras[0].time).to(means3D.device).repeat(means3D.shape[0],1)
            cur_time = ((cur_time.float() / self.T) - 0.5) * 2

            # Directly query network, used for inference
            (cur_means3D_deform, _, _, _) = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                                    rotations, opacity,
                                                                                    cur_time)

            cur_means3D_final = cur_means3D_deform
            # Warp Gaussians from obj centric to world frame
            if account_for_global_motion:
                cur_means3D_final, _ = self._objcentric2world_warp(
                    cur_means3D_final, None,
                    viewpoint_cameras[0], default_camera_center, obj_name
                )
            # Warp from virtual to real cameras
            if virtual2real is not None:
                cur_means3D_final, _ = self._virtual2real_warp(
                    cur_means3D_final, None,
                    virtual2real
                )
            
            all_new_means3D.append(cur_means3D_final)

        means3D_final = torch.cat(all_means3D_final)
        means2D = screenspace_points
        opacity = torch.cat(all_opacity)
        scales_final = torch.cat(all_scales)
        rotations_final = torch.cat(all_rotations)

        new_means3D_final = torch.cat(all_new_means3D)

        # Only update through flow, detach everything else
        rendered_3dpos_image, _, _, _, _, _ = rasterizer(
            means3D = means3D_final.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = new_means3D_final,
            opacities = opacity.detach(),
            scales = scales_final.detach(),
            rotations = rotations_final.detach(),
            cov3Ds_precomp = None
        )

        _, H_, W_ = rendered_3dpos_image.shape
        reshaped_3dpos_image = rendered_3dpos_image.permute(1,2,0).reshape(H_*W_, 3)
        rendered_2dpos_image, projected_depth = point_cloud_to_image(reshaped_3dpos_image, viewpoint_cameras[0])
        rendered_2dpos_image = rendered_2dpos_image.reshape(H_, W_, 2).permute(2, 0, 1)
        projected_depth = projected_depth.reshape(H_, W_)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "3d_pos": rendered_3dpos_image,
            "2d_pos": rendered_2dpos_image,
            "projected_depth": projected_depth
        }
    
    def _objcentric2world_warp(self, means3D_final, scales_final, viewpoint_camera, default_camera_center, obj_name):

        # Warp Gaussians from obj centric to world frame
        means3D_final = means3D_final * self.gaussian_scales[obj_name][viewpoint_camera.time]
        means3D_final = means3D_final + self.gaussian_translations[obj_name][viewpoint_camera.time]
        if scales_final is not None:
            scales_final = scales_final * self.gaussian_scales[obj_name][viewpoint_camera.time]

        # Scale Gaussians
        # Hack to deal with z conventions
        cam_center = default_camera_center.clone()
        cam_center = -cam_center
        means3D_final = cam_center - (cam_center - means3D_final) * self.gaussian_depth_scaling[obj_name][viewpoint_camera.time]
        if scales_final is not None:
            scales_final = scales_final * self.gaussian_depth_scaling[obj_name][0]

        return means3D_final, scales_final
    
    def _virtual2real_warp(self, means3D_final, rotations_final, virtual2real):

        # Warp means3D
        means3D_homo = torch.cat([means3D_final, torch.ones_like(means3D_final[:, :1])], dim=-1)
        means3D_warped = (virtual2real.to(means3D_homo.device) @ means3D_homo.T).T
        means3D_final = means3D_warped[:, :3] / means3D_warped[:, -1:]

        if rotations_final is not None:
            # Warp rotations
            rot_quat = torch.from_numpy(R.from_matrix(virtual2real[:3, :3].numpy()).as_quat()).unsqueeze(0).to(rotations_final.device)
            rot_quat = rot_quat[:, [3, 0, 1, 2]]
            rotations_from_quats = quat_mult(rot_quat, rotations_final).float()
            rotations_final = rotations_from_quats / torch.linalg.norm(rotations_from_quats, dim=-1, keepdims=True)

        return means3D_final, rotations_final

    def freeze_gaussians(self):
        for obj_name in self.gaussians.keys():
            self.gaussians[obj_name].freeze()

    def unfreeze_gaussians(self):
        for obj_name in self.gaussians.keys():
            self.gaussians[obj_name].unfreeze()
    
    def save_gaussians(self, path):
        params_dict_all_obj = {}
        for obj_name in self.gaussians.keys():
            gaussians_params = self.gaussians[obj_name].capture()
            params_dict_all_obj[obj_name] = {
                'active_sh_degree': gaussians_params[0],
                'xyz': gaussians_params[1],
                'deformation_state_dict': gaussians_params[2],
                'deformation_table': gaussians_params[3],
                'features_dc': gaussians_params[4],
                'features_rest': gaussians_params[5],
                'scaling': gaussians_params[6],
                'rotation': gaussians_params[7],
                'opacity': gaussians_params[8],
                'max_radii2D': gaussians_params[9],
                'xyz_gradient_accum': gaussians_params[10],
                'denom': gaussians_params[11],
                'optimizer_state_dict': gaussians_params[12],
                'spatial_lr_scale': gaussians_params[13],
            }

        with open(path, 'wb') as f:
            pickle.dump(params_dict_all_obj, f)
    
    def save_renderer(self, folder_path: str) -> dict:
        """
        Save the renderer state, including ply models for gaussians
        """
        os.makedirs(folder_path, exist_ok=True)
        unpickleable_keys = ['gaussians']
        state = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(value) and key not in unpickleable_keys
        }

        # save objects for reloading gaussian
        for obj_name in self.gaussians.keys():
            self.gaussians[obj_name].save_ply(os.path.join(folder_path, f'model_{obj_name}.ply'))
            self.gaussians[obj_name].save_deformation(folder_path, str(obj_name))

        with open(os.path.join(folder_path, 'renderer.pkl'), 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_renderer(cls, folder_path: str):
        """
        Load the renderer state, including ply models for gaussians
        """
        with open(os.path.join(folder_path, 'renderer.pkl'), 'rb') as f:
            state = pickle.load(f)
        renderer = cls(state['T'])
        renderer.__dict__.update(state)

        # load gaussians
        obj_names = []
        for obj_name in os.listdir(folder_path):
            if obj_name.startswith('model_') and obj_name.endswith('.ply'):
                obj_names.append(int(obj_name.split('_')[1].split('.')[0]))
        obj_names.sort()
        print(f'During load renderer, found objs: {obj_names}')
        for obj_name in obj_names:
            renderer.add_object(os.path.join(folder_path, f'model_{obj_name}.ply'), obj_name)
            renderer.gaussians[obj_name].load_model(folder_path, str(obj_name))

        return renderer

    def save_global_motion(self, path):

        params_dict = {
            'translation': self.gaussian_translation.detach(),
            'scale': self.gaussian_scale.detach(),
            'depth_scaling': self.gaussian_depth_scaling.detach(),
        }

        with open(path, 'wb') as f:
            pickle.dump(params_dict, f)

    def update_learning_rate(self, iteration):

        for param_group in self.optimizer.param_groups:
            if "xyz" in param_group["name"]:
                lr = self.gaussians[0].xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif "grid" in param_group["name"]:
                lr = self.gaussians[0].grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif "deformation" in param_group["name"]:
                lr = self.gaussians[0].deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif "translation" in param_group["name"]:
                lr = self.gaussians[0].deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif "depth_scaling" in param_group["name"]:
                lr = self.gaussians[0].deformation_scheduler_args(iteration)
                param_group['lr'] = lr
    
    def get_2d_gaussian_pos_all(self,
        viewpoint_camera,
        default_camera_center=None,
        virtual2real=None,
        account_for_global_motion=True
    ):

        all_means3D_final = []

        for obj_name in self.gaussians.keys():

            if viewpoint_camera.time in self.time_deform_T:
                idx = self.time_deform_T.index(viewpoint_camera.time)
                means3D_deform = self.means3D_deform_T[obj_name][idx]
            else:
                idx = self.prev_time_deform_T.index(viewpoint_camera.time)
                means3D_deform = self.prev_means3D_deform_T[obj_name][idx]
            
            means3D_final = means3D_deform
            # Warp Gaussians from obj centric to world frame
            if account_for_global_motion:
                means3D_final, _ = self._objcentric2world_warp(
                    means3D_final, None,
                    viewpoint_camera, default_camera_center, obj_name
                )
            # Warp from virtual to real cameras
            if virtual2real is not None:
                means3D_final, _ = self._virtual2real_warp(
                    means3D_final, None,
                    virtual2real
                )
            all_means3D_final.append(means3D_final)

        means3D_final = torch.cat(all_means3D_final)
        means2D_final, projected_depth = point_cloud_to_image(means3D_final, viewpoint_camera)

        return means2D_final, projected_depth, means3D_final
    
    @torch.no_grad()
    def get_2d_gaussian_pos_for_traj(self,
        viewpoint_camera,
        virtual2real=None,
        default_camera_center=None,
    ):
        
        all_means3D = []
        all_means2D = []
        all_projected_depth = []
        
        # Process individual object Gaussians
        for obj_name in self.gaussians.keys():

            means3D = self.gaussians[obj_name].get_xyz
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2

            opacity = self.gaussians[obj_name]._opacity
            scales = self.gaussians[obj_name]._scaling
            rotations = self.gaussians[obj_name]._rotation

            means3D_deform, _, _, _, _ = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                rotations, opacity,
                                                                time)
            
            means3D_final = means3D_deform

            # Warp Gaussians from obj centric to world frame
            means3D_final, _ = self._objcentric2world_warp(
                means3D_final, None,
                viewpoint_camera, default_camera_center, obj_name
            )
            # Warp from virtual to real cameras
            if virtual2real is not None:
                means3D_final, _ = self._virtual2real_warp(
                    means3D_final, None,
                    virtual2real
                )
            means2D_final, projected_depth = point_cloud_to_image(means3D_final, viewpoint_camera)

            all_means3D.append(means3D_final)
            all_means2D.append(means2D_final)
            all_projected_depth.append(projected_depth)
        
        means2D_final = torch.cat(all_means2D)
        projected_depth = torch.cat(all_projected_depth)
        means3D_final = torch.cat(all_means3D)

        return means2D_final, projected_depth, means3D_final

    def export_vanilla_format(self, path, t=0, max_T=1):
        """Export all gaussians to a single PLY file at time t.
        
        Args:
            path: Path to save the PLY file
            t: Time step to export at (default: 0)
            max_T: Maximum time steps (default: 1)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get deformed positions and properties for all gaussians
        all_xyz = []
        all_features_dc = []
        all_features_rest = []
        all_opacities = []
        all_scales = []
        all_rotations = []
        
        for obj_name in self.gaussians.keys():
            gaussian = self.gaussians[obj_name]
            # Get deformed properties at time t
            time = torch.tensor(t).to(gaussian.get_xyz.device).repeat(gaussian.get_xyz.shape[0], 1)
            time = ((time.float() / max_T) - 0.5) * 2
            
            means3D = gaussian.get_xyz
            scales = gaussian._scaling
            rotations = gaussian._rotation
            opacity = gaussian._opacity
            
            # Get deformed properties directly from deformation network
            means3D_deform, scales_deform, rotations_deform, opacity_deform, delta_sh = gaussian._deformation(
                means3D[gaussian._deformation_table], 
                scales[gaussian._deformation_table], 
                rotations[gaussian._deformation_table], 
                opacity[gaussian._deformation_table],
                time[gaussian._deformation_table]
            )
            
            # Initialize final tensors with original values
            means3D_final = torch.zeros_like(means3D)
            rotations_final = torch.zeros_like(rotations)
            scales_final = torch.zeros_like(scales)
            opacity_final = torch.zeros_like(opacity)
            
            # Update deformed points
            means3D_final[gaussian._deformation_table] = means3D_deform
            rotations_final[gaussian._deformation_table] = rotations_deform
            scales_final[gaussian._deformation_table] = scales_deform
            opacity_final[gaussian._deformation_table] = opacity_deform
            
            # Keep original values for non-deformed points
            means3D_final[~gaussian._deformation_table] = means3D[~gaussian._deformation_table]
            rotations_final[~gaussian._deformation_table] = rotations[~gaussian._deformation_table]
            scales_final[~gaussian._deformation_table] = scales[~gaussian._deformation_table]
            opacity_final[~gaussian._deformation_table] = opacity[~gaussian._deformation_table]
            
            # Apply global transformations if they exist
            if obj_name in self.gaussian_translations:
                means3D_final = means3D_final + self.gaussian_translations[obj_name][t]
            if obj_name in self.gaussian_scales:
                means3D_final = means3D_final * self.gaussian_scales[obj_name][t]
            
            all_xyz.append(means3D_final)
            all_features_dc.append(gaussian._features_dc)
            all_features_rest.append(gaussian._features_rest)
            all_opacities.append(opacity_final)
            all_scales.append(scales_final)
            all_rotations.append(rotations_final)
        
        # Concatenate all gaussians
        xyz = torch.cat(all_xyz, dim=0)
        features_dc = torch.cat(all_features_dc, dim=0)
        features_rest = torch.cat(all_features_rest, dim=0)
        opacities = torch.cat(all_opacities, dim=0)
        scales = torch.cat(all_scales, dim=0)
        rotations = torch.cat(all_rotations, dim=0)
        
        # Create a temporary GaussianModel without deformation network
        hyper = ModelHiddenParams(None)
        temp_gaussian = GaussianModel(self.sh_degree, hyper)
        temp_gaussian._xyz = nn.Parameter(xyz.detach())
        temp_gaussian._features_dc = nn.Parameter(features_dc.detach())
        temp_gaussian._features_rest = nn.Parameter(features_rest.detach())
        temp_gaussian._opacity = nn.Parameter(opacities.detach())
        temp_gaussian._scaling = nn.Parameter(scales.detach())
        temp_gaussian._rotation = nn.Parameter(rotations.detach())
        temp_gaussian.active_sh_degree = self.sh_degree
        
        # Save to PLY file
        temp_gaussian.save_ply(path)
