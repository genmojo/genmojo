from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')

# from sparseags.guidance_utils.zero123 import Zero123Pipeline
from zero123 import Zero123Pipeline


name_mapping = {
    "model.diffusion_model.input_blocks.1.1.": "down_blocks.0.attentions.0.",
    "model.diffusion_model.input_blocks.2.1.": "down_blocks.0.attentions.1.",
    "model.diffusion_model.input_blocks.4.1.": "down_blocks.1.attentions.0.",
    "model.diffusion_model.input_blocks.5.1.": "down_blocks.1.attentions.1.",
    "model.diffusion_model.input_blocks.7.1.": "down_blocks.2.attentions.0.",
    "model.diffusion_model.input_blocks.8.1.": "down_blocks.2.attentions.1.",
    "model.diffusion_model.middle_block.1.": "mid_block.attentions.0.",
    "model.diffusion_model.output_blocks.3.1.": "up_blocks.1.attentions.0.",
    "model.diffusion_model.output_blocks.4.1.": "up_blocks.1.attentions.1.",
    "model.diffusion_model.output_blocks.5.1.": "up_blocks.1.attentions.2.",
    "model.diffusion_model.output_blocks.6.1.": "up_blocks.2.attentions.0.",
    "model.diffusion_model.output_blocks.7.1.": "up_blocks.2.attentions.1.",
    "model.diffusion_model.output_blocks.8.1.": "up_blocks.2.attentions.2.",
    "model.diffusion_model.output_blocks.9.1.": "up_blocks.3.attentions.0.",
    "model.diffusion_model.output_blocks.10.1.": "up_blocks.3.attentions.1.",
    "model.diffusion_model.output_blocks.11.1.": "up_blocks.3.attentions.2.",
}

class Zero123(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        self.pipe = Zero123Pipeline.from_pretrained(            
            model_key,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)

        # load weights from the checkpoint
        ckpt_path = "zero123_6dof_23k.ckpt"
        print(f'[INFO] loading checkpoint from {ckpt_path} ...')
        old_state = torch.load(ckpt_path)
        pretrained_weights = old_state['state_dict']['cc_projection.weight']
        pretrained_biases = old_state['state_dict']['cc_projection.bias']
        linear_layer = torch.nn.Linear(768 + 18, 768)
        linear_layer.weight.data = pretrained_weights
        linear_layer.bias.data = pretrained_biases
        self.pipe.clip_camera_projection.proj = linear_layer.to(dtype=self.dtype, device=self.device)

        for name in list(old_state['state_dict'].keys()):
            for k, v in name_mapping.items():
                if k in name:
                    old_state['state_dict'][name.replace(k, name_mapping[k])] = old_state['state_dict'][name].to(dtype=self.dtype, device=self.device)

        m, u = self.pipe.unet.load_state_dict(old_state['state_dict'], strict=False)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]
        return c, v

    def get_cam_embeddings(self, polar, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(polar), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(polar))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(polar), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T

    def get_cam_embeddings_6D(self, target_RT, cond_RT):
        T_target = torch.from_numpy(target_RT["c2w"])
        focal_len_target = torch.from_numpy(target_RT["focal_length"])

        T_cond = torch.from_numpy(cond_RT["c2w"])
        focal_len_cond = torch.from_numpy(cond_RT["focal_length"])
    
        focal_len = focal_len_target / focal_len_cond

        d_T = torch.linalg.inv(T_target) @ T_cond
        d_T = torch.cat([d_T.flatten(), torch.log(focal_len)])
        return d_T.unsqueeze(0).unsqueeze(0).to(dtype=self.dtype, device=self.device)

    def get_cam_embeddings_6D_from_spherical(self, polar, azimuth, radius, focal_length=1.0):
        # Convert angles to radians and ensure numpy arrays
        polar_rad = np.deg2rad(polar)
        azimuth_rad = np.deg2rad(azimuth)
        
        # Handle both scalar and batch inputs
        batch_size = 1
        if isinstance(polar_rad, np.ndarray):
            batch_size = len(polar_rad)
        
        # Calculate camera position in Cartesian coordinates
        x = radius * np.cos(polar_rad) * np.sin(azimuth_rad)
        y = radius * np.cos(polar_rad) * np.cos(azimuth_rad)
        z = radius * np.sin(polar_rad)
        
        # Initialize output arrays
        c2w = np.tile(np.eye(4), (batch_size, 1, 1))
        
        # Handle each position in the batch
        for i in range(batch_size):
            # Get current position
            pos = np.array([x[i] if batch_size > 1 else x, 
                        y[i] if batch_size > 1 else y, 
                        z[i] if batch_size > 1 else z])
            
            # Calculate camera orientation
            forward = -pos / np.linalg.norm(pos)  # Look at center
            right = np.cross(np.array([0, 0, 1]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Build transformation matrix
            c2w[i, :3, :3] = np.stack([right, up, forward], axis=1)  # Rotation
            c2w[i, :3, 3] = pos  # Translation
        
        # Create target and conditioning transforms
        target_RT = {
            "c2w": c2w.astype(np.float32),
            "focal_length": np.array([focal_length] * batch_size, dtype=np.float32)
        }
        
        # For conditioning, use identity transform as reference
        cond_RT = {
            "c2w": np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32),
            "focal_length": np.array([focal_length] * batch_size, dtype=np.float32)
        }
        
        return self.get_cam_embeddings_6D(target_RT, cond_RT)

    @torch.no_grad()
    def refine(self, pred_rgb, cam_embed, 
               guidance_scale=5, steps=50, strength=0.8, idx=None
        ):

        ######## Slight modification ########
        if pred_rgb is not None:
            batch_size = pred_rgb.shape[0]
        else:
            batch_size = 1

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((1, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = cam_embed
        if idx is not None:
            cc_emb = torch.cat([self.embeddings[0][idx].repeat(batch_size, 1, 1), T], dim=-1)
        else:
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

        if idx is not None:
            vae_emb = self.embeddings[1][idx].repeat(batch_size, 1, 1, 1)
        else:
            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.view(1)]).to(self.device)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
        return imgs
    
    # def train_step(self, pred_rgb, polar, azimuth, radius, obj_name, step_ratio=None, guidance_scale=5, as_latent=False, timesteps=None, clip_loss=False):
    #     # pred_rgb: tensor [1, 3, H, W] in [0, 1]

    #     batch_size = pred_rgb.shape[0]

    #     if as_latent:
    #         latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
    #     else:
    #         pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
    #         latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

    #     if step_ratio is not None:
    #         # dreamtime-like
    #         # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
    #         t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
    #         t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
    #     else:
    #         t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

    #     w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

    #     with torch.no_grad():
    #         noise = torch.randn_like(latents)
    #         latents_noisy = self.scheduler.add_noise(latents, noise, t)

    #         x_in = torch.cat([latents_noisy] * 2)
    #         t_in = torch.cat([t] * 2)

    #         T = self.get_cam_embeddings(polar, azimuth, radius)
    #         if timesteps is not None:
    #             n_views = T.shape[0] // len(timesteps)
    #             C = self.embeddings[0][obj_name].shape[-1]
    #             cc_emb = torch.cat([self.embeddings[0][obj_name].reshape(n_views, -1, C)[:, timesteps].reshape(-1, C).unsqueeze(1), T], dim=-1)
    #         else:
    #             cc_emb = torch.cat([self.embeddings[0][obj_name].unsqueeze(1), T], dim=-1)
    #         cc_emb = self.pipe.clip_camera_projection(cc_emb)
    #         cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

    #         if timesteps is not None:
    #             n_views = T.shape[0] // len(timesteps)
    #             C = self.embeddings[1][obj_name].shape[-1]
    #             vae_emb = self.embeddings[1][obj_name].reshape(n_views, -1, 4, C, C)[:, timesteps].reshape(-1, 4, C, C)
    #         else:
    #             vae_emb = self.embeddings[1][obj_name]
    #         vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

    #         noise_pred = self.unet(
    #             torch.cat([x_in, vae_emb], dim=1),
    #             t_in.to(self.unet.dtype),
    #             encoder_hidden_states=cc_emb,
    #         ).sample

    #     noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    #     grad = w * (noise_pred - noise)
    #     grad = torch.nan_to_num(grad)

    #     target = (latents - grad).detach()
    #     loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

    #     return loss
    def train_step(self, pred_rgb, elevation, azimuth, radius, obj_name, step_ratio=None, guidance_scale=5, as_latent=False, default_elevation=0, timesteps=None, clip_loss=False):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            # T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
            T = self.get_cam_embeddings_6D_from_spherical(elevation, azimuth, radius, default_elevation)
            if timesteps is not None:
                n_views = T.shape[0] // len(timesteps)
                C = self.embeddings[0][obj_name].shape[-1]
                cc_emb = torch.cat([self.embeddings[0][obj_name].reshape(n_views, -1, C)[:, timesteps].reshape(-1, C).unsqueeze(1), T], dim=-1)
            else:
                cc_emb = torch.cat([self.embeddings[0][obj_name].unsqueeze(1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            if timesteps is not None:
                n_views = T.shape[0] // len(timesteps)
                C = self.embeddings[1][obj_name].shape[-1]
                vae_emb = self.embeddings[1][obj_name].reshape(n_views, -1, 4, C, C)[:, timesteps].reshape(-1, 4, C, C)
            else:
                vae_emb = self.embeddings[1][obj_name]
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        if clip_loss:
            pred = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            pred_pil = [TF.to_pil_image(image) for image in pred]
            pred_clip = self.pipe.feature_extractor(images=pred_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
            pred_clip_feats = self.pipe.image_encoder(pred_clip).image_embeds
            with torch.no_grad():
                if timesteps is not None:
                    C = self.embeddings[0].shape[-1]
                    clip_feats = self.embeddings[0].reshape(n_views, -1, C)[:, timesteps].reshape(-1, C)
                else:
                    clip_feats = self.embeddings[0]
            clip_score = F.cosine_similarity(pred_clip_feats, clip_feats, dim=-1)
            loss = loss + (1. - clip_score).sum()

        return loss

    def angle_between(self, sph_v1, sph_v2):
        def sph2cart(sv):
            r, theta, phi = sv[0], sv[1], sv[2]
            # The polar representation is different from Stable-DreamFusion
            return torch.tensor([r * torch.cos(theta) * torch.cos(phi), r * torch.cos(theta) * torch.sin(phi), r * torch.sin(theta)])
        def unit_vector(v):
            return v / torch.linalg.norm(v)
        def angle_between_2_sph(sv1, sv2):
            v1, v2 = sph2cart(sv1), sph2cart(sv2)
            v1_u, v2_u = unit_vector(v1), unit_vector(v2)
            return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
        angles = torch.empty(len(sph_v1), len(sph_v2))
        for i, sv1 in enumerate(sph_v1):
            for j, sv2 in enumerate(sph_v2):
                angles[i][j] = angle_between_2_sph(sv1, sv2)
        return angles

    def batch_train_step(self, pred_rgb, target_RT, cond_cams, step_ratio=None, guidance_scale=5, as_latent=False, step=None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2 * self.num_views)
            t_in = torch.cat([t] * 2 * self.num_views)

            cc_embs = []
            vae_embs = []
            noise_preds = []
            for idx in range(self.num_views):
                cond_RT = {
                    "c2w": cond_cams[idx].c2w,
                    "focal_length": cond_cams[idx].focal_length,
                }
                T = self.get_cam_embeddings_6D(target_RT, cond_RT)
                cc_emb = torch.cat([self.embeddings[0][idx].repeat(batch_size, 1, 1), T], dim=-1)
                cc_emb = self.pipe.clip_camera_projection(cc_emb)
                cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

                vae_emb = self.embeddings[1][idx].repeat(batch_size, 1, 1, 1)
                vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

                cc_embs.append(cc_emb)
                vae_embs.append(vae_emb)

            cc_emb = torch.cat(cc_embs, dim=0)
            vae_emb = torch.cat(vae_embs, dim=0)
            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

            noise_pred_chunks = noise_pred.chunk(self.num_views)
            for idx in range(self.num_views):
                noise_pred_cond, noise_pred_uncond = noise_pred_chunks[idx][0], noise_pred_chunks[idx][1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_preds.append(noise_pred)

        noise_pred = torch.stack(noise_preds).sum(dim=0) / len(noise_preds) # self.num_views # Average over all views

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        return loss

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents