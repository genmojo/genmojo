import torch
import torch.nn.functional as F
import numpy as np
import tqdm

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ

def run_flow_on_images(model, images,
                    padding_factor=32,
                    inference_size=None,
                    attn_splits_list=(2,),
                    corr_radius_list=(-1,),
                    prop_radius_list=(-1,)
                    ):

    model.eval()

    stride = 1

    fwd_flows = []
    bwd_flows = []
    fwd_valids = []
    bwd_valids = []

    for test_id in range(0, len(images) - 1, stride):

        image1 = (images[test_id] * 255)
        image2 = (images[test_id+1] * 255)

        if inference_size is None:
            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()

        # resize before inference
        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             pred_bidir_flow=True,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
        assert flow_pr.size(0) == 2  # [2, H, W, 2]

        # resize back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if inference_size is None:
            fwd_flow = padder.unpad(flow_pr[0])  # [2, H, W,]
        else:
            fwd_flow = flow_pr[0]  # [2, H, W,]

        # also predict backward flow
        if inference_size is None:
            bwd_flow = padder.unpad(flow_pr[1])  # [2, H, W,]
        else:
            bwd_flow = flow_pr[1]  # [2, H, W,]

        fwd_flows.append(fwd_flow)
        bwd_flows.append(bwd_flow)
        
        # forward-backward consistency check
        # occlusion is 1
        fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow.unsqueeze(0), bwd_flow.unsqueeze(0))  # [1, H, W] float

        fwd_valids.append(1. - fwd_occ)
        bwd_valids.append(1. - bwd_occ)

    return fwd_flows, bwd_flows, fwd_valids, bwd_valids

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


import cv2

def run_tracker_on_images(model, images, binary_mask):  # 0 for dense track
    # import pdb; pdb.set_trace()
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    # import pdb; pdb.set_trace()
    # Ensure binary mask is correctly thresholded
    # _, binary_mask = cv2.threshold(binary_mask, 127, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask[0, 0].cpu().numpy()

    kernel = np.ones((2, 2), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # import pdb; pdb.set_trace()

    # Find the coordinates of all foreground points (non-zero points in binary mask)
    foreground_coords = np.column_stack(np.where(binary_mask > 0))

    # Determine the number of points to sample
    num_foreground_points = len(foreground_coords)
    num_samples = min(10000, num_foreground_points) # ori is 10000

    assert num_foreground_points > 0
    # Randomly sample points if more than 10,000 foreground points
    if num_foreground_points > 10000: # ori is 10000
        sampled_indices = np.random.choice(num_foreground_points, size=num_samples, replace=False)
        sampled_coords = foreground_coords[sampled_indices]
    else:
        sampled_coords = foreground_coords
    
    # import pdb; pdb.set_trace()
    # Create a new binary mask with sampled points only
    sampled_binary_mask = np.zeros_like(binary_mask)
    sampled_binary_mask[sampled_coords[:, 1], sampled_coords[:, 0]] = 1



    # Get foreground points' (x, y) coordinates
    foreground_points = np.column_stack(np.where(sampled_binary_mask == 1))  # shape (N, 2), N is the number of points
     
    # import pdb; pdb.set_trace()

    # Convert to PyTorch tensor and reshape to (1, N, 2)
    foreground_points_tensor = torch.tensor(foreground_points, dtype=torch.float32).unsqueeze(0)  # shape (1, N, 2)

    # Concatenate zero dimension for queries
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    queries = torch.cat([torch.zeros_like(foreground_points_tensor[:, :, :1]), foreground_points_tensor], dim=2).to(device)  # shape (1, N, 3)
    # num_samples = 10000
    # indices = torch.randperm(queries.shape[1])[:num_samples]
    # import pdb; pdb.set_trace()

    # queries = queries[:,indices,:]

    window_frames = []
    
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries = queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
    
    # import pdb; pdb.set_trace()

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(images):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=100, # will not be used
                grid_query_frame=0,
            )
            is_first_step = False
        
        # import pdb; pdb.set_trace()

        frame = frame * 255.
        
        window_frames.append(frame)
        # print('model step:', model.step)
    

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=10,
        grid_query_frame=0,
    )

    # import pdb; pdb.set_trace()
    
    if False:
        import pdb; pdb.set_trace()
        from cotracker.utils.visualizer import Visualizer
        seq_name = 'test_video.mp4'
        video = torch.tensor(np.stack(window_frames), device=device).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(
            video, pred_tracks, pred_visibility, query_frame=0
        )

    T = len(window_frames)
    H, W = 512, 512

    flow_map = torch.zeros(T, H, W, 2).to(device)
    valid_masks = torch.zeros(T, H, W).to(device)
    

    flow_map[:, sampled_coords[:, 0], sampled_coords[:, 1],:] = pred_tracks[0]
    valid_masks[:, sampled_coords[:, 0], sampled_coords[:, 1]] = pred_visibility[0].float()

    flow_map = flow_map - flow_map[0:1]
    #valid_masks = valid_masks - valid_masks[0:1]

    flow_map = flow_map[1:]
    valid_masks = valid_masks[1:]

    return flow_map.permute(0, 3, 1, 2), valid_masks  # T-1, 2, H, W and T-1, H, W
