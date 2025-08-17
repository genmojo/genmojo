# Copyright (c) 2024 GenMOJO and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import glob
import tqdm
import json
import math
import scipy
import pickle
import numpy as np

import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from torchmetrics import PearsonCorrCoef
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cameras import orbit_camera, OrbitCamera, MiniCam
from utils.general_utils import safe_normalize
from gs_renderer_4d_fs import Renderer
from utils.flow_utils import run_flow_on_images, run_tracker_on_images
from utils.loss_utils import local_pearson_loss, pearson_depth_loss
from gmflow.gmflow.gmflow import GMFlow

from PIL import Image
from torchvision.transforms.functional import center_crop

from diffusers.utils import export_to_gif
from scipy.spatial.transform import Rotation as R
from utils.traj_visualizer import Visualizer

from grid_put import mipmap_linear_grid_put_2d
import cv2
from PIL import Image
from utils.feature_utils import dino_per_pixel_features, load_dino

@torch.no_grad()
def get_filtered_gaussians_mask(gaussian_2d_pos, gaussian_depth, depth, alpha):

    H, W = depth.shape
    gaussian_x = gaussian_2d_pos[:, 0].round().int()
    gaussian_y = gaussian_2d_pos[:, 1].round().int()

    # Check visibility of gaussians
    visible = (alpha > 0.75) # or maybe 0.5?
    # Remove regions close to the boundary
    visible = scipy.ndimage.binary_erosion(visible, structure=np.ones((7, 7))).astype(visible.dtype)
    visible_mask = visible[torch.clamp(gaussian_y, 0, H-1), torch.clamp(gaussian_x, 0, W-1)]
    # Check if gaussians are within bounds
    within_bounds = ((gaussian_x > 0) & (gaussian_x < W-1) & (gaussian_y > 0) & (gaussian_y < H-1)).numpy()
    # Only consider front facing gaussians
    
    # import pdb; pdb.set_trace()

    init_rendered_depth = depth[torch.clamp(gaussian_y, 0, H-1), torch.clamp(gaussian_x, 0, W-1)]

    lower_bound = init_rendered_depth * 0.9
    upper_bound = init_rendered_depth * 1.1

    # Determine which values in gaussian_depth_se are within the 10% range of the rendered depth
    matched_depth_mask = (gaussian_depth >= lower_bound) & (gaussian_depth <= upper_bound)
    # import pdb; pdb.set_trace()
    # matched_depth_mask = np.abs(gaussian_depth - init_rendered_depth) < 0.25 # or maybe 0.01?

    filter_mask = (within_bounds * visible_mask * matched_depth_mask).astype(bool)

    return filter_mask

def get_render_depth_map(gaussian_2d_pos, alpha_map):
    gaussian_x = gaussian_2d_pos[:, :, 0].round().long()
    gaussian_y = gaussian_2d_pos[:, :, 1].round().long()

    # Clamp to ensure the indices are within valid image boundaries (0 to 511)
    gaussian_x = torch.clamp(gaussian_x, min=0, max=511)
    gaussian_y = torch.clamp(gaussian_y, min=0, max=511)

    T, N = gaussian_x.shape
    batch_indices = torch.arange(T).view(T, 1).expand(T, N)  # [T, N]

    # Extract the alpha values for each point [T, N]
    visible_map = alpha_map[batch_indices, 0, gaussian_y, gaussian_x]  # [T, N]
    visible_map = torch.from_numpy(visible_map).unsqueeze(-1)

    return visible_map

@torch.no_grad()
def match_gaussians_to_pixels(gaussian_2d_pos, gaussian_depth, gt_2d_pos, rendered_depth, rendered_alpha):

    # This function matches pixels to Gaussians

    gaussian_2d_pos = torch.from_numpy(gaussian_2d_pos)
    gt_2d_pos = torch.from_numpy(gt_2d_pos)

    init_gaussian_2d_pos = gaussian_2d_pos[0]
    init_gt_2d_pos = gt_2d_pos[0]
    init_gaussian_depth = gaussian_depth[0]
    init_depth = rendered_depth[0].squeeze(0)
    init_alpha = rendered_alpha[0].squeeze(0)

    # Get relevant Gaussians
    filter_mask = get_filtered_gaussians_mask(init_gaussian_2d_pos, init_gaussian_depth, init_depth, init_alpha)
    init_gaussian_2d_pos_filtered = init_gaussian_2d_pos[filter_mask]

    # Find matching between GT and Gaussians
    dist_to_gaussians = torch.norm(torch.unsqueeze(init_gt_2d_pos, 1)-torch.unsqueeze(init_gaussian_2d_pos_filtered, 0), dim=-1)
    min_dist_to_gaussians, min_idxes = torch.min(dist_to_gaussians, dim=1)
    dist_thresh = 0.25
    while dist_thresh < 4:
        dist_thresh = dist_thresh * 2
        matched_mask = min_dist_to_gaussians < dist_thresh

    matched_idxes = min_idxes[matched_mask]
    query_points = gt_2d_pos[0][matched_mask]

    render_depth_sel = get_render_depth_map(gaussian_2d_pos[:, filter_mask][:, matched_idxes], rendered_depth)
    gaussian_depth_se = torch.from_numpy(gaussian_depth[:, filter_mask][:, matched_idxes]).unsqueeze(-1)
    lower_bound = render_depth_sel * 0.9
    upper_bound = render_depth_sel * 1.1

    # Determine which values in gaussian_depth_se are within the 10% range of the rendered depth
    gaussian_occ_mask = (gaussian_depth_se >= lower_bound) & (gaussian_depth_se <= upper_bound)

    return gaussian_2d_pos[:, filter_mask][:, matched_idxes], gaussian_occ_mask, query_points

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.seed = 888

        # models
        self.device = torch.device("cuda")

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None

        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False
        
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5
        
        self.depth_scale = 1.0

        self.obj_name_list = [i for i in range(opt.obj_num)]
        self.obj_input_dict = {}
        for obj_name in self.obj_name_list:

            self.obj_input_dict[obj_name] = {
                "input_img": None,
                "input_depth": None,
                "input_depth_mask": None,
                "input_mask": None,
                "input_img_torch": None,
                "input_depth_torch": None,
                "input_depth_mask_torch": None,
                "input_mask_torch": None,
                
                "input_img_list": None,
                "input_depth_list": None,
                "input_depth_mask_list": None,
                "input_mask_list": None,
                "input_img_torch_list": None,
                "input_depth_torch_list": None,
                "input_depth_mask_torch_list": None,
                "input_mask_torch_list": None
            }

        self.vid_length = None
        
        self.flow_model = 'gmflow'
        # depth model
        if self.opt.depth_model is not None:
            assert self.opt.depth_model in ['depthcrafter', 'depthanything'], f"Depth model {self.opt.depth_model} not supported"
            self.depth_model = self.opt.depth_model
        else:
            self.depth_model = 'depthanything'
            print(f"[WARNING] Depth model not specified, using {self.depth_model}")

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        # load input data from cmdline
        for obj_name in self.obj_name_list:
            self.load_input(self.opt.input, obj_name) # load imgs, if has bg, then rm bg; or just load imgs
        self.get_depth(self.opt.input, self.obj_name_list)
        
        # renderer
        self.renderer = Renderer(T=self.vid_length, sh_degree=self.opt.sh_degree)
        self.render_obj_dict = {}

        if self.opt.cam_path is not None:
            self.load_cam_poses(self.opt.cam_path)
        else:
            self.real_cam_poses = []
            self.virtual2reals = []
            default_pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            for _ in range(self.vid_length):
                self.real_cam_poses.append(default_pose)
                self.virtual2reals.append(torch.eye(4))
        
        # override if provide a checkpoint
        if self.opt.load is not None:
            for obj_name in self.obj_name_list:
                file_name_load = f'{self.opt.outdir}/{opt.save_path}_{obj_name+1}_model.ply'
                self.renderer.add_object(file_name_load, obj_name)
        else:
            assert(False)
        
        self.enlarge_factor = 1.0 # ori is 1.0
        for obj_name in self.obj_name_list:
            self.scale_depth(obj_name)

        self.seed_everything()
        
    @torch.no_grad()
    def scale_depth(self, obj_name):
            
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.obj_input_dict[obj_name]['scale_fact'] = []

        print(f"Finding depth scale for obj {obj_name}")

        for t in tqdm.tqdm(range(len(self.obj_input_dict[obj_name]['input_depth_list']))):

            cur_cam = MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far
            )
            cur_cam.time = t

            with torch.no_grad():
                out = self.renderer.render_all(
                    viewpoint_camera=cur_cam,
                    specify_obj=obj_name,
                    account_for_global_motion=False,
                    no_deform=True,
                    direct_render=True
                )
            rendered_depth = out["depth"].squeeze().cpu() # [H, W]
            rendered_depth = torch.nan_to_num(rendered_depth)

            # Erode and remove invalid parts of the mask
            obj_mask = out["alpha"].squeeze().cpu() > 0.75
            obj_mask = obj_mask.float().numpy()
            eroded_mask = scipy.ndimage.binary_erosion(obj_mask > 0.5, structure=np.ones((7, 7)))

            if eroded_mask.sum() > 0:
                rendered_depth = torch.median(rendered_depth[eroded_mask])
                input_depth = self.obj_input_dict[obj_name]['input_depth_list'][t]
                input_depth_mask = self.obj_input_dict[obj_name]['input_depth_mask_list'][t] > 0.5
                metric_depth = torch.median(torch.from_numpy(input_depth[input_depth_mask]))
                self.obj_input_dict[obj_name]['rendered_depth'] = rendered_depth

                if obj_name == 0 and t == 0:
                    self.depth_scale = (rendered_depth / metric_depth).item()
                    self.obj_input_dict[obj_name]['scale_fact'].append(torch.tensor(1.0))
                else:
                    depth_scale_factor = (self.depth_scale * metric_depth) / rendered_depth
                    if torch.isnan(depth_scale_factor):
                        depth_scale_factor = self.obj_input_dict[obj_name]['scale_fact'][-1]
                    self.obj_input_dict[obj_name]['scale_fact'].append(depth_scale_factor)

            else:
                self.obj_input_dict[obj_name]['scale_fact'].append(self.obj_input_dict[obj_name]['scale_fact'][-1])

            depth = self.obj_input_dict[obj_name]['input_depth_list'][t].copy()
            scaled_depth = depth * self.depth_scale * self.enlarge_factor
            self.obj_input_dict[obj_name]['input_depth_list'][t] = scaled_depth

        def gaussian_kernel(size: int, sigma: float):
            x = torch.arange(-size//2 + 1, size//2 + 1).float()
            kernel = torch.exp(-x**2 / (2 * sigma**2))
            return kernel / kernel.sum()

        kernel = gaussian_kernel(5, 1.0).view(1, 1, -1)
        padded_scales = F.pad(torch.stack(self.obj_input_dict[obj_name]['scale_fact']).unsqueeze(0).unsqueeze(0), (5 // 2, 5 // 2), mode="replicate")
        smoothed_scales = F.conv1d(padded_scales, kernel).squeeze()
        scales_diff = torch.diff(smoothed_scales)
        scales_diff = torch.clamp(scales_diff, -0.05, 0.05)
        smoothed_scales = torch.cat([torch.tensor([smoothed_scales[0]]), smoothed_scales[0] + torch.cumsum(scales_diff, dim=0)])

        self.obj_input_dict[obj_name]['smoothed_depth_scales'] = torch.tensor(smoothed_scales, dtype=torch.float32, device="cuda")

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.set_detect_anomaly(mode=False)

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # do not do progressive sh-level
        for obj_name in self.renderer.gaussians.keys():
            self.renderer.gaussians[obj_name].active_sh_degree = self.renderer.gaussians[obj_name].max_sh_degree

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 # False
        self.enable_zero123 = self.opt.lambda_zero123 > 0 # True

        # lazy load guidance model

        if self.guidance_zero123 is None and self.enable_zero123: # True
            from guidance.zero123_per_obj_utils import Zero123
            if self.opt.stable_zero123:
                print(f"[INFO] loading stable zero123...")
                self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max], model_key='ashawkey/stable-zero123-diffusers')
            else:
                print(f"[INFO] loading zero123...")
                self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max], model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
        
        for obj_name in self.obj_name_list:
            # Load frame 0 scales
            with open(os.path.join(self.opt.outdir, "gaussians", str(self.opt.save_path)+'_'+str(obj_name + 1)+"_global_motion.pkl"), 'rb') as f:
                self.obj_input_dict[obj_name]['input_scale0'] = pickle.load(f)['scale'].squeeze()

        # Parse images and masks
        if self.obj_input_dict[obj_name]['input_img_list'] is not None:
            
            if self.opt.feature_splatting:
                print('using dino feature splatting')
                dino = load_dino('cuda')

            height, width = self.obj_input_dict[obj_name]['input_img_list'][0].shape[:2]
            resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
            H_ = int(height * resize_factor)
            W_ = int(width * resize_factor)
            
            for obj_name in self.obj_name_list:
                self.obj_input_dict[obj_name]['input_img_torch_obj_centric_list'] = []
                self.obj_input_dict[obj_name]['input_mask_torch_obj_centric_list'] = []

                self.obj_input_dict[obj_name]['input_img_torch_orig_list'] = []
                self.obj_input_dict[obj_name]['input_mask_torch_orig_list'] = []
                self.obj_input_dict[obj_name]['input_depth_torch_orig_list'] = []
                self.obj_input_dict[obj_name]['input_depth_mask_torch_orig_list'] = []

                self.obj_input_dict[obj_name]['obj_cx_list'] = []
                self.obj_input_dict[obj_name]['obj_cy_list'] = []
                self.obj_input_dict[obj_name]['obj_scale_list'] = []
                
                self.obj_input_dict[obj_name]['input_feat_list'] = []
                # import pdb; pdb.set_trace()
                for idx, (input_img, input_img_ori, input_mask, input_depth, input_depth_mask) in enumerate(zip(self.obj_input_dict[obj_name]['input_img_list'], self.obj_input_dict[obj_name]['input_img_list_ori'], self.obj_input_dict[obj_name]['input_mask_list'], self.obj_input_dict[obj_name]['input_depth_list'], self.obj_input_dict[obj_name]['input_depth_mask_list'])):
                    # Reshape
                    input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                    input_img_torch_ori = torch.from_numpy(input_img_ori).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                    input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                    input_depth_torch = torch.from_numpy(input_depth).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                    input_depth_mask_torch = torch.from_numpy(input_depth_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                     
                    # Resize world frame inputs to 720p
                    self.obj_input_dict[obj_name]['input_img_torch_orig_list'].append(F.interpolate(input_img_torch_ori, (H_, W_), mode="bilinear", align_corners=False))
                    self.obj_input_dict[obj_name]['input_mask_torch_orig_list'].append(F.interpolate(input_mask_torch, (H_, W_), mode="nearest"))
                    self.obj_input_dict[obj_name]['input_depth_torch_orig_list'].append(F.interpolate(input_depth_torch, (H_, W_), mode="nearest"))
                    self.obj_input_dict[obj_name]['input_depth_mask_torch_orig_list'].append(F.interpolate(input_depth_mask_torch, (H_, W_), mode="nearest"))
                    # print(self.obj_input_dict[obj_name]['input_img_torch_orig_list'][0].shape, '4d image shape', len(self.obj_input_dict[obj_name]['input_img_torch_orig_list']))
                    # torch.Size([1, 3, 512, 512]) 4d image shape 1, 2, ..., 40
                    if self.opt.feature_splatting:
                        img_for_dino = self.obj_input_dict[obj_name]['input_img_torch_orig_list'][-1]           # [1,3,256,256]
                        pix_feats = dino_per_pixel_features(img_for_dino, dino)
                        # print('dino feature shape', pix_feats.shape)  # [1,384,256,256]
                        self.obj_input_dict[obj_name]['input_feat_list'].append(pix_feats)
                    N, C, H, W = input_mask_torch.shape

                    mask = input_mask_torch > 0.5
                    nonzero_idxes = torch.nonzero(mask[0,0])
                    if len(nonzero_idxes) > 0:
                        # Find bbox
                        min_x = nonzero_idxes[:, 1].min()
                        max_x = nonzero_idxes[:, 1].max()
                        min_y = nonzero_idxes[:, 0].min()
                        max_y = nonzero_idxes[:, 0].max()
                        # Find cx cy
                        cx = (max_x + min_x) / 2
                        cx = ((cx / W) * 2 - 1)
                        cy = (max_y + min_y) / 2
                        cy = ((cy / H) * 2 - 1)
                        self.obj_input_dict[obj_name]['obj_cx_list'].append(cx)
                        self.obj_input_dict[obj_name]['obj_cy_list'].append(cy)
                        # Find maximum possible scale
                        width = (max_x - min_x) / W
                        height = (max_y - min_y) / H
                        scale_x = width / 0.9
                        scale_y = height / 0.9
                        scale = max(scale_x, scale_y)
                        # If the scale from the first frame doesn't clip the object, then stick with it, otherwise use the max possible scale
                        scale = max(scale, self.obj_input_dict[obj_name]['input_scale0']) #HACK: TODO
                        self.obj_input_dict[obj_name]['obj_scale_list'].append(scale)

                        # Construct affine warp and grid
                        theta = torch.tensor([[[scale, 0., cx], [0., scale, cy]]], device=self.device)
                        resize_factor = self.opt.ref_size / min(H, W)
                        grid = F.affine_grid(theta, (N, C, int(H*resize_factor), int(W*resize_factor)), align_corners=True)
                        # Change border of image to white because we assume white background
                        input_img_torch_obj_centric = input_img_torch.clone()
                        input_img_torch_obj_centric[:, :, 0] = 1.
                        input_img_torch_obj_centric[:, :, -1] = 1.
                        input_img_torch_obj_centric[:, :, :, 0] = 1.
                        input_img_torch_obj_centric[:, :, :, -1] = 1.
                        # Aspect preserving grid sample, this recenters and scales the object
                        input_img_torch_obj_centric = F.grid_sample(input_img_torch_obj_centric, grid, align_corners=True, padding_mode='border')
                        input_mask_torch_obj_centric = F.grid_sample(input_mask_torch.clone(), grid, align_corners=True)
                        # Center crop
                        input_img_torch_obj_centric = center_crop(input_img_torch_obj_centric, self.opt.ref_size)
                        input_mask_torch_obj_centric = center_crop(input_mask_torch_obj_centric, self.opt.ref_size)

                        self.obj_input_dict[obj_name]['input_img_torch_obj_centric_list'].append(input_img_torch_obj_centric)
                        self.obj_input_dict[obj_name]['input_mask_torch_obj_centric_list'].append(input_mask_torch_obj_centric)
                    else:
                        # Empty masks, use dummy values, will be ignored in rendering losses
                        print(f'[Warning] empty mask found, append with last value')
                        self.obj_input_dict[obj_name]['obj_cx_list'].append(self.obj_input_dict[obj_name]['obj_cx_list'][-1])
                        self.obj_input_dict[obj_name]['obj_cy_list'].append(self.obj_input_dict[obj_name]['obj_cy_list'][-1])
                        self.obj_input_dict[obj_name]['obj_scale_list'].append(self.obj_input_dict[obj_name]['obj_scale_list'][-1])
                        self.obj_input_dict[obj_name]['input_img_torch_obj_centric_list'].append(self.obj_input_dict[obj_name]['input_img_torch_obj_centric_list'][-1])
                        self.obj_input_dict[obj_name]['input_mask_torch_obj_centric_list'].append(self.obj_input_dict[obj_name]['input_mask_torch_obj_centric_list'][-1])
            
            for obj_name in self.obj_name_list:
                # prepare flow
                self.obj_input_dict[obj_name]['input_flow_torch_orig_list'] = []
                self.obj_input_dict[obj_name]['input_flow_valid_torch_orig_list'] = []
                with torch.no_grad():
                    if self.flow_model == 'gmflow':
                        # Load GMFlow
                        flow_predictor = GMFlow(
                            feature_channels=128,
                            num_scales=1,
                            upsample_factor=8,
                            num_head=1,
                            attention_type='swin',
                            ffn_dim_expansion=4,
                            num_transformer_layers=6,
                            attn_splits_list=[2],
                            corr_radius_list=[-1],
                            prop_radius_list=[-1],
                        )
                        flow_predictor.eval()
                        checkpoint = torch.load(self.opt.gmflow_path)
                        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
                        flow_predictor.load_state_dict(weights)
                        flow_predictor.to(self.device, non_blocking=True)
                        # Run GMFlow
                        fwd_flows_orig, _, fwd_valids_orig, _ = run_flow_on_images(flow_predictor, torch.cat(self.obj_input_dict[obj_name]['input_img_torch_orig_list']))
                        
                        del flow_predictor

                    elif self.flow_model == 'cotracker':
                        # Load cotracker
                        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)

                        # Run cotracker
                        fwd_flows_orig, fwd_valids_orig = run_tracker_on_images(cotracker, torch.cat(self.obj_input_dict[obj_name]['input_img_torch_orig_list']), self.obj_input_dict[obj_name]['input_mask_torch_orig_list'][0])

                        del cotracker
                    else:
                        raise TypeError
                
                # Since there's no frame -1, for frame 0, we set a flow map of all zeros
                self.obj_input_dict[obj_name]['input_flow_torch_orig_list'].append(torch.zeros_like(fwd_flows_orig[0].unsqueeze(0)))
                self.obj_input_dict[obj_name]['input_flow_valid_torch_orig_list'].append(torch.ones_like(fwd_flows_orig[0].unsqueeze(0)))
                # Mask out (set to zero) irrelevant parts of the flow using previous frame masks
                # For valid masks, 1 if it's within the object and passed consistency check
                for i, (flow_orig, valid_orig) in enumerate(zip(fwd_flows_orig, fwd_valids_orig)):
                    self.obj_input_dict[obj_name]['input_flow_torch_orig_list'].append((flow_orig.unsqueeze(0)).clone())
                    self.obj_input_dict[obj_name]['input_flow_valid_torch_orig_list'].append((valid_orig.unsqueeze(0)).clone()) 
            
        torch.cuda.empty_cache()
        
        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = {}, {}
                for obj_name in self.obj_name_list:
                    c_list_obj, v_list_obj = [], []
                    for view_idx in range(self.opt.n_views):
                        for idx, obj_centric_img in enumerate(self.obj_input_dict[obj_name]['input_img_torch_obj_centric_list']):
                            c, v = self.guidance_zero123.get_img_embeds(obj_centric_img)
                            c_list_obj.append(c)
                            v_list_obj.append(v)

                    c_list[obj_name] = torch.cat(c_list_obj, 0)
                    v_list[obj_name] = torch.cat(v_list_obj, 0)
                        
                self.guidance_zero123.embeddings = [c_list, v_list]

            if self.enable_svd:
                assert(False)
                self.guidance_svd.get_img_embeds(self.input_img)

        # prepare global warps
        for obj_name in self.obj_name_list:

            # Figure out how much we need to shift in 3D space based on displacements in the screen space
            render_height, render_width = self.obj_input_dict[obj_name]['input_img_torch_orig_list'][0].shape[-2:]
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
            cam = MiniCam(pose, render_width, render_height, render_cam.fovy, render_cam.fovx, render_cam.near, render_cam.far)
            median_z = torch.median(self.renderer.gaussians[obj_name].get_xyz[:, -1]).detach()
            dist_to_cam = render_cam.campos[-1] - median_z
            x_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[0, 0]
            y_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[1, 1]
            
            # Initialize the warp
            translations = []
            for obj_cx, obj_cy in zip(self.obj_input_dict[obj_name]['obj_cx_list'], self.obj_input_dict[obj_name]['obj_cy_list']):
                translation = torch.tensor(
                    [obj_cx*x_scale, -obj_cy*y_scale, 0.],
                    dtype=torch.float32,
                    device="cuda",
                )
                translations.append(translation)
                
            translations = torch.stack(translations)
            scales = torch.stack(self.obj_input_dict[obj_name]['obj_scale_list']) * self.enlarge_factor # newly added
            smoothed_depth_scales = self.obj_input_dict[obj_name]['smoothed_depth_scales']

            depth_scales = torch.tensor(
                smoothed_depth_scales,
                dtype=torch.float32,
                device="cuda",
            )
            self.renderer.init_obj_warps(translations, scales, depth_scales, obj_name)

        self.renderer.create_joint_optimizer(self.opt, 0.1)
        self.optimizer = self.renderer.optimizer

    def get_cam_at_timestep(self, timestep, render_width=None, render_height=None, cam_param_obj=None):

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        # If we don't specify anything, use default cam
        if self.opt.cam_path is None and render_width is None and render_height is None and cam_param_obj is None:
            cam = copy.deepcopy(self.fixed_cam)
        # Otherwise we need to construct a custom cam
        else:
            # Replace camera pose if specified
            if self.opt.cam_path is not None:
                pose = self.real_cam_poses[timestep]
            # Replace render params if specified
            if render_width is None:
                render_width = self.opt.ref_size
            if render_height is None:
                render_height = self.opt.ref_size
            if cam_param_obj is None:
                cam_param_obj = self.cam
            # Construct cam object using custom params
            cam = MiniCam(
                pose,
                render_width,
                render_height,
                cam_param_obj.fovy,
                cam_param_obj.fovx,
                cam_param_obj.near,
                cam_param_obj.far,
            )
        # Finally, set the timestep for the cam
        cam.time = timestep
        
        # Return the virtual2real transform, identity if not specified
        virtual2real = self.virtual2reals[timestep]

        return cam, pose, virtual2real
    
    def train_step(self):

        # Use custom cam params for rendering world frame RGB
        render_height, render_width = self.obj_input_dict[0]['input_img_torch_orig_list'][0].shape[-2:]
        render_cam_world = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)

        for _ in range(self.train_steps):
            self.step += 1
            self.optimizer.zero_grad(set_to_none=True)
            # perform gradient accumulation
            for accum_step in range(self.opt.grad_accumulation_step):
                step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500

                # update lr
                self.renderer.update_learning_rate(self.step)
                loss = 0

                rand_timesteps = np.random.choice(np.arange(len(self.obj_input_dict[0]['input_img_torch_orig_list'])), self.opt.batch_size, replace=False).tolist()
                self.renderer.prepare_render_all(rand_timesteps)
            
                ### known view
                for i, b_idx in enumerate(rand_timesteps):
                    
                    cur_cam, pose, virtual2real = self.get_cam_at_timestep(b_idx, render_width, render_height, render_cam_world)
                    bg_color = torch.tensor(
                        [1, 1, 1] if np.random.random() < 0.5 else [0, 0, 0],
                        dtype=torch.float32,
                        device="cuda",
                    )

                    out = self.renderer.render_all(
                        cur_cam,
                        virtual2real=virtual2real,
                        default_camera_center=self.fixed_cam.camera_center,
                        bg_color=bg_color,
                        direct_render=False,
                        cut_gaussians=False
                    )

                    # rgb loss
                    if self.opt.feature_splatting:
                        feat_splat = out["feat_map"].unsqueeze(0) # [1, C, H, W] in [0, 1]
                        loss = loss + 10000 * step_ratio * F.mse_loss(feat_splat, self.obj_input_dict[0]['input_feat_list'][b_idx])
                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    target_img = self.obj_input_dict[0]['input_img_torch_orig_list'][b_idx]
                    
                    sum_mask = torch.zeros_like(self.obj_input_dict[0]['input_mask_torch_orig_list'][b_idx] > 0.5).float()
                    target_instance_mask = torch.zeros_like(self.obj_input_dict[0]['input_mask_torch_orig_list'][b_idx] > 0.5).float()
                    for obj_idx, obj_name in enumerate(self.obj_name_list):
                        sum_mask += (self.obj_input_dict[obj_name]['input_mask_torch_orig_list'][b_idx] > 0.5).float()
                        target_instance_mask += (self.obj_input_dict[obj_name]['input_mask_torch_orig_list'][b_idx] > 0.5).float() * (obj_idx+1) # 0 is BG, obj starts from 1
                    target_mask = (sum_mask > 0.5).float()
                    target_img = target_img * target_mask + (1 - target_mask) * bg_color.reshape(1, 3, 1, 1)
                    if target_mask.sum() > 0:
                        loss = loss + 10000 * step_ratio * self.balanced_rgb_loss(image, target_img, target_mask) / self.opt.batch_size

                    # mask loss
                    mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                    if target_mask.sum() > 0:
                        loss = loss + 1000 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask) / self.opt.batch_size

                    instance_mask = out["extra"].unsqueeze(0)
                    #instance_mask /= (instance_mask.sum(1).unsqueeze(1) + 1e-6)
                    if step_ratio > 0.75:
                        N, H, W = instance_mask[0].shape
                        probs = instance_mask[0].permute(1, 2, 0).reshape(-1, N)  # [H * W, N]
                        target = torch.clamp(sum_mask-1, min=0).squeeze(1).long().view(-1)  # [H * W]
                        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1).reshape(1, H, W)

                        nonoverlap_mask = (sum_mask <= 1.).float()
                        target_instance_mask = target_instance_mask * nonoverlap_mask
                        class_loss = F.nll_loss(torch.log(instance_mask+1e-10), torch.clamp(target_instance_mask-1, min=0).squeeze(1).long(), reduction='none')
                        class_loss_valid_mask = ((target_mask * nonoverlap_mask).squeeze(1) > 0.5)*(target_probs < 0.75)
                        if class_loss_valid_mask.sum() > 0:
                            class_loss = class_loss[class_loss_valid_mask].mean() # mask out bg
                            loss = loss + 100 * step_ratio * class_loss / self.opt.batch_size
                    
                    # Render flow in the context of prev frame
                    prev_cam, _, prev_virtual2real = self.get_cam_at_timestep(b_idx - 1 if b_idx > 0 else 0, render_width, render_height, render_cam_world)
                    out = self.renderer.render_flow_all(
                        [cur_cam, prev_cam],
                        virtual2real=prev_virtual2real,
                        default_camera_center=self.fixed_cam.camera_center,
                        bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                        account_for_global_motion=True
                    )
                        
                    # flow loss
                    _, H, W = out["flow"].shape
                    flow = out["flow"].permute(1,2,0).reshape((-1, 3)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H, W, 2)).permute(2,0,1)
                    flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    
                    target_flow = self.obj_input_dict[0]['input_flow_torch_orig_list'][b_idx]
                    prev_sum_mask = torch.zeros_like(self.obj_input_dict[0]['input_mask_torch_orig_list'][b_idx] > 0.5).float()
                    obj_flow_valid_mask = torch.zeros_like(self.obj_input_dict[0]['input_flow_valid_torch_orig_list'][b_idx])
                    for obj_name in self.obj_name_list:
                        obj_flow_valid_mask += self.obj_input_dict[obj_name]['input_flow_valid_torch_orig_list'][b_idx]
                        if b_idx > 0:
                            prev_sum_mask += (self.obj_input_dict[obj_name]['input_mask_torch_orig_list'][b_idx-1] > 0.5).float()
                    prev_mask = (prev_sum_mask > 0.5).float()
                    target_flow = target_flow * prev_mask
                    
                    obj_flow_valid_mask = (obj_flow_valid_mask > 0.5).float() 
                    flow_loss = (flow_2d - target_flow).abs()

                    # We will normalize the flow loss w.r.t. to image dimensions
                    flow_loss[:, 0] /= W
                    flow_loss[:, 1] /= H
                    # Note that this is different from the normal balanced mask loss
                    if prev_mask.sum() > 0:
                        if (prev_mask * obj_flow_valid_mask).sum() > 0:
                            masked_flow_loss = (flow_loss * prev_mask * obj_flow_valid_mask).sum() / (prev_mask * obj_flow_valid_mask).sum()
                        else:
                            masked_flow_loss = 0.
                        if (1 - prev_mask).sum() > 0:
                            masked_flow_loss_empty = (flow_loss * (1 - prev_mask)).sum() / (1 - prev_mask).sum()
                        else:
                            masked_flow_loss_empty = 0.
                        loss = loss + 10000. * step_ratio * (masked_flow_loss + masked_flow_loss_empty) / self.opt.batch_size

                    # reg losses
                    reg_loss = out["scale_change"].abs().mean()
                    loss = loss + 1000. * step_ratio * reg_loss / self.opt.batch_size

                    reg_sh_loss = out["sh_change"].abs().mean()
                    loss = loss + 500. * step_ratio * reg_sh_loss / self.opt.batch_size

                    local_scale_loss = out["local_scale_loss"]
                    loss = loss + 50000. * step_ratio * local_scale_loss / self.opt.batch_size

                    local_rigidity_loss = out["local_rigidity_loss"]
                    loss = loss + 50000. * step_ratio * local_rigidity_loss / self.opt.batch_size

                # scale loss by accumulation steps
                loss = loss / self.opt.grad_accumulation_step
                loss.backward()

                if self.opt.do_guidance_step:
                    ### novel view (manual batch)
                    for rand_obj_name in self.obj_name_list:
                        loss = 0
                        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
                        images = []
                        poses = []
                        vers, hors, radii = [], [], []
                        # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
                        min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
                        max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

                        self.renderer.prepare_render_all(rand_timesteps, specify_obj=rand_obj_name)

                        for view_idx in range(self.opt.n_views+1):
                            for b_idx in rand_timesteps:

                                if view_idx == 0:
                                    # render obj centric view
                                    pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
                                    cur_cam = MiniCam(
                                        pose,
                                        self.opt.ref_size,
                                        self.opt.ref_size,
                                        self.cam.fovy,
                                        self.cam.fovx,
                                        self.cam.near,
                                        self.cam.far
                                    )
                                    cur_cam.time = b_idx
                                    bg_color = torch.tensor(
                                        [1, 1, 1] if np.random.random() < 0.5 else [0, 0, 0],
                                        dtype=torch.float32,
                                        device="cuda",
                                    )

                                    out = self.renderer.render_all(
                                        cur_cam,
                                        specify_obj=rand_obj_name,
                                        bg_color=bg_color,
                                        direct_render=False,
                                        account_for_global_motion=False,
                                    )

                                    guidance_loss = 0
                                    # rgb loss
                                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                                    target_img = self.obj_input_dict[rand_obj_name]['input_img_torch_obj_centric_list'][b_idx]
                                    
                                    target_mask = torch.zeros_like(self.obj_input_dict[rand_obj_name]['input_mask_torch_obj_centric_list'][b_idx] > 0.5).float()
                                    target_mask = (target_mask > 0.5).float()
                                    target_img = target_img * target_mask + (1 - target_mask) * bg_color.reshape(1, 3, 1, 1)
                                    if target_mask.sum() > 0:
                                        fg_rgb_loss = (F.mse_loss(image, target_img, reduction='none') * target_mask).sum() / target_mask.sum()
                                        guidance_loss = guidance_loss + 1000 * step_ratio * fg_rgb_loss / self.opt.batch_size

                                    # mask loss
                                    mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                                    if target_mask.sum() > 0:
                                        fg_mask_loss = (F.mse_loss(mask, target_mask, reduction='none') * target_mask).sum() / target_mask.sum()
                                        guidance_loss = guidance_loss + 1000 * step_ratio * fg_mask_loss / self.opt.batch_size
                                
                                    loss = loss + guidance_loss
                                else:

                                    # render random view
                                    ver = np.random.randint(min_ver, max_ver)
                                    hor = np.random.randint(-180, 180)
                                    radius = 0

                                    vers.append(ver)
                                    hors.append(hor)
                                    radii.append(radius)

                                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                                    poses.append(pose)

                                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, time=b_idx)

                                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                                    if self.enable_zero123:
                                        # Render one obj at obj centric frame
                                        out = self.renderer.render_all(
                                            cur_cam,
                                            specify_obj=rand_obj_name,
                                            cut_gaussians=False,
                                            bg_color=bg_color, 
                                            direct_render=False,
                                            account_for_global_motion=False)
                                    else:
                                        assert(False)

                                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                                    images.append(image)

                        images = torch.cat(images, dim=0)

                        # guidance loss
                        if self.enable_zero123:
                            zero123_loss = self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, rand_obj_name, step_ratio, timesteps=rand_timesteps) / (self.opt.batch_size * self.opt.n_views)
                            loss = loss + zero123_loss

                        # scale loss by accumulation steps
                        loss = loss / self.opt.grad_accumulation_step
                        loss.backward()

                if (accum_step + 1) % self.opt.grad_accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

    def balanced_rgb_loss(self, pred, target, mask):
        # Compute loss over mask and non-mask regions separately
        if mask.sum() > 0:
            masked_loss = (F.mse_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
        else:
            masked_loss = 0.
        if (1 - mask).sum() > 0:
            masked_loss_empty = (F.mse_loss(pred, target, reduction='none') * (1 - mask)).sum() / (1 - mask).sum()
        else:
            masked_loss_empty = 0.

        return masked_loss + masked_loss_empty
    
    def balanced_mask_loss(self, pred, target, mask):
        # Compute loss over mask and non-mask regions separately
        if mask.sum() > 0:
            masked_loss = (F.mse_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
        else:
            masked_loss = 0.
        if (1 - mask).sum() > 0:
            masked_loss_empty = (F.mse_loss(pred, target, reduction='none') * (1 - mask)).sum() / (1 - mask).sum()
        else:
            masked_loss_empty = 0.

        return masked_loss + masked_loss_empty

    def load_input(self, file, obj_name):
        print("Loading input images and masks")
        assert(self.opt.input_mask is not None)
        # Get file lists
        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        file_list = file_list
        self.vid_length = len(file_list)
        mask_file_list = glob.glob(self.opt.input_mask[obj_name]+'*' if self.opt.input_mask[obj_name][-1] == '/' else self.opt.input_mask[obj_name]+'/*')
        mask_file_list.sort()
        mask_file_list = mask_file_list

        self.obj_input_dict[obj_name]['input_img_list'], self.obj_input_dict[obj_name]['input_mask_list'] = [], []
        self.obj_input_dict[obj_name]['input_img_list_ori'] = []

        # Load files
        for i, file in enumerate(tqdm.tqdm(file_list)):
            img = Image.open(file)
            if self.opt.resize_square:
                width, height = img.size
                new_dim = min(width, height)
                img = img.resize([new_dim, new_dim], Image.Resampling.BICUBIC)
            img = np.array(img)[:, :, :3]
            mask = Image.open(mask_file_list[i])
            if self.opt.resize_square:
                mask = mask.resize([new_dim, new_dim], Image.Resampling.NEAREST)
            mask = np.array(mask)
            mask = mask.astype(np.float32) / 255.0
            if len(mask.shape) == 3:
                mask = mask[:, :, 0:1]
            else:
                mask = mask[:, :, np.newaxis]

            img = img.astype(np.float32) / 255.0
            input_mask = mask
            # white bg
            input_img = img[..., :3] * input_mask + (1 - input_mask)
            self.obj_input_dict[obj_name]['input_img_list'].append(input_img)
            self.obj_input_dict[obj_name]['input_img_list_ori'].append(img[..., :3])
            self.obj_input_dict[obj_name]['input_mask_list'].append(input_mask)

    @torch.no_grad()
    def get_depth(self, file, obj_name_list):

        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        file_list = file_list

        for obj_name in obj_name_list:
            self.obj_input_dict[obj_name]['input_depth_list'], self.obj_input_dict[obj_name]['input_depth_mask_list'] = [], []

        print(f"Extracting depth with {self.depth_model}")
        if self.depth_model == 'depthanything':

            image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf")
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf")
            model.eval()
        
            for i, file in enumerate(tqdm.tqdm(file_list)):
                image = Image.open(file)
                
                with torch.no_grad():
                    # prepare image for the model
                    inputs = image_processor(images=image, return_tensors="pt")
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth

                # interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                depth = prediction.squeeze().numpy()

                for obj_name in obj_name_list:
                    eroded_mask = scipy.ndimage.binary_erosion(self.obj_input_dict[obj_name]['input_mask_list'][i].squeeze(-1) > 0.5, structure=np.ones((7, 7)))
                    eroded_mask = (eroded_mask > 0.5)

                    masked_depth = depth * eroded_mask

                    self.obj_input_dict[obj_name]['input_depth_list'].append(masked_depth[:, :, np.newaxis].astype(np.float32))
                    self.obj_input_dict[obj_name]['input_depth_mask_list'].append(eroded_mask[:, :, np.newaxis].astype(np.float32))
            
            del model
            torch.cuda.empty_cache()

        elif self.depth_model == 'depthcrafter':

            from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
            from DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                "tencent/DepthCrafter",
                torch_dtype=torch.float16,
            )
            pipe = DepthCrafterPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
        
            # depth crafter can operate on all frames
            frames = []
            for file in tqdm.tqdm(file_list):
                image = Image.open(file)
                frames.append(image)
            frames = np.stack(frames) / 255

            with torch.no_grad():
                res_pipe = pipe(
                    frames,
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type="np",
                    num_inference_steps=5,
                    guidance_scale=1.0,
                    window_size=110,
                    overlap=25,
                    track_time=False
                )
                res = res_pipe.frames[0]
            # convert the three-channel output to a single channel depth map
            res = res.sum(-1) / res.shape[-1]
            # invert the depth map
            depth_all_frames = 1 / res

            for i, file in enumerate(tqdm.tqdm(file_list)):
                for obj_name in obj_name_list:
                    eroded_mask = scipy.ndimage.binary_erosion(self.obj_input_dict[obj_name]['input_mask_list'][i].squeeze(-1) > 0.5, structure=np.ones((7, 7)))
                    eroded_mask = (eroded_mask > 0.5)

                    masked_depth = depth_all_frames[i, ...] * eroded_mask

                    self.obj_input_dict[obj_name]['input_depth_list'].append(masked_depth[:, :, np.newaxis].astype(np.float32))
                    self.obj_input_dict[obj_name]['input_depth_mask_list'].append(eroded_mask[:, :, np.newaxis].astype(np.float32))
            
            del unet, pipe
            torch.cuda.empty_cache()
        
        else:
            raise NotImplementedError(f"Depth model {self.depth_model} not supported")
            
    def load_cam_poses(self, file):

        self.real_cam_poses = []
        self.virtual2reals = []

        loaded_poses = np.load(file)['cam_c2w']

        for t in range(self.vid_length):

            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = pose[:3, :3].T
            w2c[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
            # Loaded pose is x: right, y: in, z: up
            # We need x: right, y: up, z: out
            axis_convert = np.array([
                [1.,  0.,  0.,  0.],
                [0.,  0.,  1.,  0.],
                [0., -1.,  0.,  0.],
                [0.,  0.,  0.,  1.]
            ])

            # Load original camera pose
            loaded_pose = loaded_poses[t]
            real_cam_pose = (axis_convert @ loaded_pose).astype(np.float32)

            # Flip relative rotation
            if t > 0:
                rel_rot = real_cam_pose[:3, :3] @ self.real_cam_poses[0][:3, :3].T
                real_cam_pose[:3, :3] = rel_rot.T @ self.real_cam_poses[0][:3, :3]
            virtual2real = torch.from_numpy(real_cam_pose @ w2c)

            self.real_cam_poses.append(real_cam_pose)
            self.virtual2reals.append(virtual2real)

    def render_visualization(self, file_name, hor = 180, specify_obj=None, render='rgb', account_for_global_motion=False):

        assert render in ['rgb', 'mask', 'depth']

        render_height, render_width = self.obj_input_dict[0]['input_img_torch_orig_list'][0].shape[-2:]
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)

        image_list = []
        nframes = self.vid_length * 5
        # hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                render_width,
                render_height,
                render_cam.fovy,
                render_cam.fovx,
                render_cam.near,
                render_cam.far,
                time=time
            )
            with torch.no_grad():
                outputs = self.renderer.render_all(
                    cur_cam,
                    default_camera_center=self.fixed_cam.camera_center if account_for_global_motion else None,
                    direct_render=True,
                    specify_obj=specify_obj,
                    account_for_global_motion=account_for_global_motion,
                    cut_gaussians= True
                )

            if render == 'mask':
                out = outputs["extra"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
            elif render == 'depth':
                ALPHA_THRESH = 0.95
                out = outputs["depth"].squeeze(0).cpu().detach().numpy().astype(np.float32)
                depth_mask = outputs["alpha"].squeeze(0).cpu().detach().numpy().astype(np.float32)
                out -= out[depth_mask > ALPHA_THRESH].min()
                out /= (out[depth_mask > ALPHA_THRESH].max() - out[depth_mask > ALPHA_THRESH].min())
                out = out * (depth_mask > 0.75)
            else:
                out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.vid_length

        print('export to GIF')
        export_to_gif(image_list, file_name)

    def render_demo_rotating(self, file_name, hor_range = [170, 190], elev_range = [-10, 10], radius = 2, specify_obj=None, account_for_global_motion=False):
        render_height, render_width = self.obj_input_dict[0]['input_img_torch_orig_list'][0].shape[-2:]
        render_cam = OrbitCamera(render_width, render_height, r=radius, fovy=self.opt.fovy)

        image_list = []
        nframes = self.vid_length * 5
        # Calculate delta_hor for smooth oscillation
        total_angle = (hor_range[1] - hor_range[0])
        delta_hor = 2 * total_angle / nframes  # Double the angle change to account for back and forth
        time = 0
        delta_time = 1
        direction = 1  # 1 for moving right, -1 for moving left
        current_hor = hor_range[0]

        total_elev_angle = (elev_range[1] - elev_range[0])
        delta_elev = 2 * total_elev_angle / nframes
        elev_direction = 1
        current_elev = elev_range[0]

        for _ in range(nframes):
            pose = orbit_camera(current_elev, current_hor-180, radius)
            cur_cam = MiniCam(
                pose,
                render_width,
                render_height,
                render_cam.fovy,
                render_cam.fovx,
                render_cam.near,
                render_cam.far,
                time=time
            )
            with torch.no_grad():
                outputs = self.renderer.render_all(
                    cur_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                    direct_render=True,
                    specify_obj=specify_obj,
                    account_for_global_motion=account_for_global_motion,
                    cut_gaussians=True
                )

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.vid_length
            
            # Update horizontal angle with oscillation
            current_hor += direction * delta_hor
            # Change direction when reaching bounds
            if current_hor >= hor_range[1]:
                current_hor = hor_range[1]
                direction = -1
            elif current_hor <= hor_range[0]:
                current_hor = hor_range[0]
                direction = 1
            
            current_elev += elev_direction * delta_elev
            if current_elev >= elev_range[1]:
                current_elev = elev_range[1]
                elev_direction = -1
            elif current_elev <= elev_range[0]:
                current_elev = elev_range[0]
                elev_direction = 1

        print('export to GIF')
        export_to_gif(image_list, file_name)
    
    @torch.no_grad()
    def collect_gaussian_trajs(self, orbit_angle, elevation_angle):

        # Get Gaussian motion
        rendered_rgb = []
        rendered_alpha = []
        rendered_depth = []
        gaussian_2d_pos = []
        gaussian_3d_pos = []
        gaussian_depth = []

        for t in range(self.vid_length):
            pose = orbit_camera(elevation_angle, orbit_angle, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=t
            )
            with torch.no_grad():
                outputs = self.renderer.render_all(
                    viewpoint_camera=cur_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                    direct_render=True,
                    account_for_global_motion=True,
                    cut_gaussians= True,
                )
                rendered_rgb.append(outputs["image"].cpu().numpy().astype(np.float32))
                rendered_depth.append(outputs["depth"].cpu().numpy())
                rendered_alpha.append(outputs["alpha"].cpu().numpy())
                pos_2d, proj_depth, pos_3d = self.renderer.get_2d_gaussian_pos_for_traj(cur_cam, default_camera_center=self.fixed_cam.camera_center)

            gaussian_2d_pos.append(pos_2d.cpu().numpy())
            gaussian_3d_pos.append(pos_3d.cpu().numpy())
            gaussian_depth.append(proj_depth.cpu().numpy())
        
        gaussian_2d_pos = np.stack(gaussian_2d_pos)
        gaussian_3d_pos = np.stack(gaussian_3d_pos)
        gaussian_depth = np.stack(gaussian_depth)
        rendered_depth = np.stack(rendered_depth)
        rendered_alpha = np.stack(rendered_alpha)
        rendered_rgb = np.stack(rendered_rgb)

        return gaussian_2d_pos, gaussian_3d_pos, gaussian_depth, rendered_depth, rendered_alpha, rendered_rgb

    @torch.no_grad()
    def render_gaussian_trajs(self, file_name, orbit_angle=0, elevation_angle=0):

        # First collect 2d Gaussian trajs
        gaussian_2d_pos, gaussian_3d_pos, gaussian_depth, rendered_depth, rendered_alpha, rendered_rgb = self.collect_gaussian_trajs(orbit_angle, elevation_angle)
 
        sum_mask = torch.zeros_like(self.obj_input_dict[0]['input_mask_torch_orig_list'][0] > 0.5).float()
        for obj_name in self.obj_name_list:
            sum_mask += (self.obj_input_dict[obj_name]['input_mask_torch_orig_list'][0] > 0.5).float()
        target_mask = (sum_mask > 0.5).float() 
        
        # Sample points on a grid in the mask
        merged_mask = target_mask
        merged_mask = np.transpose(rendered_alpha[0] > 0.75, (1,2,0))
        merged_mask = torch.from_numpy(rendered_alpha[0] > 0.75)[None].float()
        merged_mask = F.interpolate(
            merged_mask,
            (512, 512),
            mode="nearest"
        ).squeeze().bool().cpu().detach().numpy()
        H, W = merged_mask.shape[:2]
        center = [H / 2, W / 2]
        margin = W / 64
        range_y = (margin - H / 2 + center[0], H / 2 + center[0] - margin)
        range_x = (margin - W / 2 + center[1], W / 2 + center[1] - margin)
        grid_y, grid_x = np.meshgrid(
            np.linspace(*range_y, 48),
            np.linspace(*range_x, 48),
            indexing="ij",
        )
        grid_pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        point_mask = merged_mask[
            (grid_pts[:, 1]).astype(int),
            (grid_pts[:, 0]).astype(int)
        ].astype(bool)
        grid_pts = grid_pts[point_mask][np.newaxis]

        # Find matching gaussians to these "query points"
        gaussian_2d_traj, gaussian_2d_traj_visible, query_points = match_gaussians_to_pixels(gaussian_2d_pos, gaussian_depth, grid_pts, rendered_depth, rendered_alpha)
        
        # Visualization
        vis = Visualizer(
            save_dir=self.opt.visdir,
            linewidth=2,
            mode='rainbow',
            tracks_leave_trace=10,
        )
        # Load images
        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()

        video = []
        for i, file in enumerate(file_list):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = img[:, :, :3]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA) #* 0.4
            video.append(img[:, :, ::-1])
        # Plot trajs
        vis.visualize(
            # video=torch.from_numpy(np.stack(video)).permute(0,3,1,2).unsqueeze(0),
            video=torch.from_numpy(rendered_rgb*255).unsqueeze(0),
            tracks=gaussian_2d_traj.unsqueeze(0),
            visibility=gaussian_2d_traj_visible.unsqueeze(0),
            filename=file_name)
        np.save(os.path.join(self.opt.visdir, file_name+".npy"), query_points)

    def save_state(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        # save gui states
        unpickleable_keys = ['renderer']
        state = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(value) and key not in unpickleable_keys
        }

        # save renderer
        self.renderer.save_renderer(folder_path)

        with open(os.path.join(folder_path, 'gui.pkl'), 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved GUI state to folder {folder_path}")

    @classmethod
    def load_state(cls, opt, folder_path: str):
        with open(os.path.join(folder_path, 'gui.pkl'), 'rb') as f:
            state = pickle.load(f)
        
        gui = cls(state['opt'])
        gui.__dict__.update(state)
        # should use new opt instead of saved one
        gui.opt = opt
        # load renderer and gaussians
        renderer = gui.renderer.load_renderer(folder_path)
        gui.renderer = renderer
        
        return gui

    # no gui mode
    def train(self, iters):
        # import pdb; pdb.set_trace()
        # Main training loop    
        if iters > 0:
            self.prepare_train()
    
        # Joint finetune
        self.step = 0
        for _ in tqdm.trange(int(iters)):
            self.train_step()

        # Save
        self.save_state(os.path.join(self.opt.outdir, self.opt.save_path, "gui_states"))

        self.visualize()
    
    def visualize(self):
        # Render eval
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}.gif'),
            account_for_global_motion=True,
        )
        self.render_demo_rotating(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_rotating.gif'), 
            hor_range=[170, 190], 
            elev_range=[-20, 0], 
            radius=2.5, 
            account_for_global_motion=True
        )

        self.render_gaussian_trajs(file_name=f'{self.opt.save_path}_trajs', orbit_angle=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    opt.load = os.path.join(opt.outdir, opt.save_path + '_1_model.ply')
    os.makedirs(opt.visdir, exist_ok=True)

    if opt.visualize_only:
        state_path = os.path.join(opt.outdir, opt.save_path, "gui_states")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State files not found at {state_path}")
        gui = GUI.load_state(opt, state_path)
        gui.visualize()
    else:
        gui = GUI(opt)
        gui.train(opt.iters)
