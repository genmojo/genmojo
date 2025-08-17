import os
import argparse
import glob
import subprocess
        
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help="Path to RGB directory.")
parser.add_argument('--mask_dir', type=str, help="Path to mask directory.")
parser.add_argument('--save_name', type=str, help="Identifier for the video, determines filenames that will be saved.")
parser.add_argument('--out_dir', type=str, default="./gaussians", help="Path to where Gaussians and network weights will be stored.")
parser.add_argument('--vis_dir', type=str, default="./vis", help="Path to where visualizations will be stored.")
parser.add_argument("--lite", action="store_true", help="Uses less optimization iterations. Much faster, but may result in worse results.") 
parser.add_argument("--disable_guidance", action="store_true", help="Disables SDS for guidance.") 

if __name__ == "__main__":
    
    
    args = parser.parse_args()
        
    num_objs = len([x[1] for x in os.walk(args.mask_dir)][0])

    for obj_idx in range(1, num_objs):
        vid_dir = os.path.join(args.data_dir)
        mask_dir = os.path.join(args.mask_dir)
        frames = glob.glob(vid_dir + '/*')
        frames.sort()
        frame0 = frames[0]
        frame0_name = os.path.basename(frame0)

        load_path = os.path.join(args.out_dir, "gaussians", f"{args.save_name}_{obj_idx}")

        command = f'python main_fs.py --config configs/image.yaml input={frame0} input_mask={os.path.join(args.mask_dir, str(obj_idx).zfill(3), frame0_name)} outdir={args.out_dir} visdir={args.vis_dir} save_path={args.save_name}_{obj_idx}'
        subprocess.Popen(command, shell=True).wait()
    
    vid_dir = os.path.join(args.data_dir)
    frames = glob.glob(vid_dir + '/*')
    frames.sort()
    frame0 = frames[0]
    frame0_name = os.path.basename(frame0)
    first_frame_masks = []
    for obj_idx in range(1, num_objs):
        first_frame_masks.append(os.path.join(args.mask_dir, str(obj_idx).zfill(3)))
    first_frame_masks = '[{}]'.format(','.join(map(str, first_frame_masks)))

    train_iters = len(frames) * 30 if args.lite else len(frames) * 40
    do_guidance_step = not args.disable_guidance
    
    depth_model = 'depthcrafter'
    command = f'python main_4d_fs.py --config configs/4d.yaml depth_model={depth_model} input={args.data_dir} input_mask={first_frame_masks} outdir={args.out_dir} visdir={args.vis_dir} save_path={args.save_name} iters={train_iters} depth_loss=True obj_num={num_objs-1} batch_size=8 grad_accumulation_step=2 feature_splatting=True do_guidance_step={do_guidance_step}'
    subprocess.Popen(command, shell=True).wait()