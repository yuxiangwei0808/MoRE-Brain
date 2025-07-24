import os
import torch
import numpy as np

root = 'evals/2/Text+Image-mse/subj2-40sess-frozen_exp_cproj-MoEMulti2-noMeta-TimeRouterAttn'
split = 'val'
files = os.listdir(root)
 
files = [f for f in files if split in f and 'temp' not in f]
# files = [f for f in files if split in f]
files_indices = [f for f in files if 'indices' in f]
files_recons = [f for f in files if 'recons' in f]
files_clip_txt = [f for f in files if 'clip_txt' in f]
files_clip_img = [f for f in files if 'clip_img' in f]

assert len(files_indices) == len(files_recons) == 8

files_indices.sort()
files_recons.sort()

indices = [np.load(os.path.join(root, f)).flatten() for f in files_indices]
recons = [torch.load(os.path.join(root, f), map_location='cpu') for f in files_recons]


indices = np.concatenate(indices, 0)
recons = [torch.cat(r, dim=0) for r in recons]
recons = torch.cat(recons, dim=0)

# remove repeated samples
_, unique_indices = np.unique(indices, return_index=True)
indices = indices[unique_indices]
recons = recons[unique_indices]

np.save(os.path.join(root, f'all_indices_{split}.npy'), indices)
torch.save(recons, os.path.join(root, f'all_recons_{split}.pt'))

# delete original files
for f in files_indices + files_recons:
    os.remove(os.path.join(root, f))

if files_clip_img:
    files_clip_img.sort()
    clip_img = [np.load(os.path.join(root, f)) for f in files_clip_img]
    try:
        clip_img = np.concatenate(clip_img, axis=0)[unique_indices]
        np.save(os.path.join(root, f'all_clip_img_{split}.npy'), clip_img)
    except Exception:
        ...
    for f in files_clip_img:
        os.remove(os.path.join(root, f))
if files_clip_txt:
    files_clip_txt.sort()
    clip_txt = [np.load(os.path.join(root, f)) for f in files_clip_txt]
    try:
        clip_txt = np.concatenate(clip_txt, axis=0)[unique_indices]
        np.save(os.path.join(root, f'all_clip_txt_{split}.pt'), clip_txt)
    except Exception:
        ...
    for f in files_clip_txt:
        os.remove(os.path.join(root, f))