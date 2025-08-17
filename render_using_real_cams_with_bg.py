# Copyright (c) 2024 GenMOJO and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import scipy
import tqdm
import pickle
import numpy as np
import json
import copy

import argparse
from omegaconf import OmegaConf

import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_gif
from transformers import pipeline, AutoProcessor, CLIPModel
from scipy.spatial.transform import Rotation as R

from cameras import orbit_camera, OrbitCamera, MiniCam
from gs_renderer_4d import Renderer
from utils.traj_visualizer import Visualizer

class GUI:
    def __init__(self, opt):

        self.opt = opt

        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
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

        self.seed = 888

        self.seed_everything()
        
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

    def scale_bg_depth(self):
        bg_w, bg_h = self.bg_depth.shape[0], self.bg_depth.shape[1]
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        cur_cam = MiniCam(
            pose,
            bg_w,
            bg_h,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far
        )
        cur_cam.time = 0  # Use first frame

        with torch.no_grad():
            out = self.renderer.render_all(
                viewpoint_camera=cur_cam,
                specify_obj='background',
                account_for_global_motion=False,
                no_deform=True,
                direct_render=True
            )
        rendered_depth = out["depth"].squeeze().cpu()
        rendered_depth = torch.nan_to_num(rendered_depth)
        rendered_depth = rendered_depth[rendered_depth > 0]

        # Get mask for background
        obj_mask = out["alpha"].squeeze().cpu() > 0.75
        obj_mask = obj_mask.float().numpy()
        eroded_mask = scipy.ndimage.binary_erosion(obj_mask > 0.5, structure=np.ones((7, 7)))

        if eroded_mask.sum() > 0:
            rendered_depth = torch.median(rendered_depth)
            metric_depth = torch.median(torch.from_numpy(self.bg_depth))

            # Calculate depth scale factor same way as foreground
            depth_scale_factor = (self.depth_scale * metric_depth) / rendered_depth

            # Scale background depth
            self.bg_depth_scale_factor = depth_scale_factor * self.enlarge_factor

        print(f"Background depth scale factor: {self.bg_depth_scale_factor}")
    
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

    def render_visualization_real_cam(self, file_name, orbit=None, depth=False):

        assert orbit in [None]

        render_height, render_width = self.obj_input_dict[0]['input_img_torch_orig_list'][0].shape[-2:]
        
        image_list = []
        nframes = self.vid_length
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        render_cam_world = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
        cur_cam_t0, _, virtual2real_t0 = self.get_cam_at_timestep(0, render_width, render_height, render_cam_world)
        for t_idx in range(nframes):

            cur_cam, pose, virtual2real = self.get_cam_at_timestep(t_idx, render_width, render_height, render_cam_world)

            with torch.no_grad():
                outputs = self.renderer.render_all(
                    cur_cam,
                    virtual2real=virtual2real,
                    virtual2real_t0=virtual2real_t0,
                    default_camera_center=self.fixed_cam.camera_center,
                    bg_color=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
                    account_for_global_motion=True,
                    direct_render=True,
                    cut_gaussians=True
                )
                out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
                out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size

        export_to_gif(image_list, file_name)

    def render_demo_rotating(self, file_name, hor_range = [170, 190], elev_range = [-10, 10], radius = 2, specify_obj=None, account_for_global_motion=True):
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

    @classmethod
    def load_state(cls, opt, folder_path: str):
        with open(os.path.join(folder_path, 'gui.pkl'), 'rb') as f:
            state = pickle.load(f)
        
        gui = cls(state['opt'])
        gui.__dict__.update(state)
        # should use new opt instead of saved one
        gui.opt = opt
        # load renderer and gaussians
        gui.vid_length = state['vid_length']
        renderer = Renderer.load_renderer(folder_path)
        gui.renderer = renderer
        # process bg if any
        if opt.bg_input is not None:
            gui.renderer.add_object(opt.bg_input, 'background', init_zero_deformation=True)
            gui.renderer.gaussians["background"].freeze()
            # load depthmap
            gui.bg_depth = np.load(os.path.join(gui.opt.outdir, "gaussians", str(gui.opt.save_path)+"_bg_depth.npy"))
            # scale depth to same as foreground
            gui.scale_bg_depth()
            # init obj warp for background
            gui.renderer.init_obj_warps(
                torch.zeros((gui.vid_length, 3), device="cuda"),
                torch.ones((gui.vid_length, 1), device="cuda"),
                torch.ones((gui.vid_length, 1), device="cuda") * gui.bg_depth_scale_factor,
                "background",
            )
        # load real cam
        if opt.cam_path is not None:
            gui.load_cam_poses(opt.cam_path)
        else:
            gui.real_cam_poses = []
            gui.virtual2reals = []
            default_pose = orbit_camera(gui.opt.elevation, 0, gui.opt.radius)
            for _ in range(gui.vid_length):
                gui.real_cam_poses.append(default_pose)
                gui.virtual2reals.append(torch.eye(4))

        return gui

    def visualize(self):
        # render real cam
        self.render_visualization_real_cam(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_real_cam.gif'),
        )

        self.render_demo_rotating(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_rotating.gif'), 
            hor_range=[170, 190], 
            elev_range=[-20, 0], 
            radius=2.5, 
            account_for_global_motion=True
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    opt.load = os.path.join(opt.outdir, opt.save_path + '_1_model.ply')
    os.makedirs(opt.visdir, exist_ok=True)

    state_path = os.path.join(opt.outdir, opt.save_path, "gui_states")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"State files not found at {state_path}")
    gui = GUI.load_state(opt, state_path)

    gui.visualize()