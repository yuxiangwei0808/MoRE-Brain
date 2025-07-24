import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
import webdataset as wds
import h5py
import random
from tqdm import tqdm


def load_nsd(num_sessions, subj, num_devices, batch_size, test_batch_size, multi_subject=False):
    data_path = 'data/NSD'
    def my_split_by_node(urls): return urls
    if multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != subj]
    else:
        subj_list = [subj]

    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])

    if multi_subject:
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        num_samples_per_epoch = (750*40) // num_devices 
    else:
        num_samples_per_epoch = 750*num_sessions // num_devices 

    batch_size = batch_size // len(subj_list)
    num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

    ###### load train data ######
    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    num_voxels_list = []
    
    for s in subj_list:
        if multi_subject:
            train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        else:
            train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
        print(train_url)
        
        train_set = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                            .shuffle(750, initial=1500, rng=random.Random(42))\
                            .decode("torch")\
                            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])

        train_data[f'subj0{s}'] = train_set
        train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)

        f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(torch.float16)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

    train_dls = {f'subj0{s}': train_dl[f'subj0{s}'] for s in subj_list}

    #### Load Test ########
    class CustomDataset(Dataset):
        def __init__(self, images, voxels, indices, first_300=False):
            self.images = images
            self.voxels = voxels
            self.indices = indices
            if first_300:
                self.images = self.images[:300]
                self.voxels = self.voxels[:300]
                self.indices = self.indices[:300]
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'voxel': self.voxels[idx],
                'img_idx': self.indices[idx]
            }

    if multi_subject: 
        subj = subj_list[0] # cant validate on the actual held out person so picking first in subj_list
    test_dataset = np.load(f'{data_path}/processed/test_data_{subj}.npz', allow_pickle=True)['arr_0'][()]
    test_gt = torch.Tensor(test_dataset['image'])
    test_voxel = torch.Tensor(test_dataset['voxel'])
    test_indices = test_dataset['image_idx']

    test_dataset = CustomDataset(test_gt, test_voxel, test_indices)
    test_dl = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=12, pin_memory=True)

    return train_dls, voxels, batch_size, test_dl, num_voxels_list, subj_list, num_iterations_per_epoch


def load_train_data(train_dls, voxels, subj_list, accelerator, batch_size, num_devices, num_sessions, num_iterations_per_epoch, data_type=torch.float16, data_path='data/NSD'):
    handle = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
    all_gt = handle['images']

    image_iters, voxel_iters, image_idx_iters = {s: [] for s in subj_list}, {s: [] for s in subj_list}, {s: [] for s in subj_list}
    with accelerator.autocast():
        for subj, train_dl in zip(subj_list, train_dls):
            for it, (behav0, past_behav, future_behav, old_behav) in enumerate(train_dl):
                image_idx = behav0[:, 0, 0].cpu().long().numpy()
                image0 = torch.stack([torch.tensor(all_gt[i], dtype=data_type) for i in image_idx])

                voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                voxel0 = voxels[f'subj0{subj}'][voxel_idx]
                voxel0 = torch.Tensor(voxel0).unsqueeze(1)

                image_iters[subj].append(image0)
                voxel_iters[subj].append(voxel0)
                image_idx_iters[subj].append(image_idx)

                if it >= num_iterations_per_epoch-1:
                    break
    
    return image_iters, voxel_iters, image_idx_iters


if __name__ == '__main__':
    from accelerate import Accelerator
    # generate_test_set()
    train_dls, voxels, batch_size, test_dl, num_voxels_list, subj_list, num_iterations_per_epoch = load_nsd(40, 1, 1, 32, 32)
    image_iters, voxel_iters, image_idx_iters = load_train_data(train_dls.values(), voxels, subj_list, Accelerator(), batch_size, 1, 40, num_iterations_per_epoch)

    subj = 1
    all_images, all_voxels, all_indices = [], [], []
    for it in range(num_iterations_per_epoch):
        all_images.append(image_iters[subj][it])
        all_voxels.append(voxel_iters[subj][it])
        all_indices.append(image_idx_iters[subj][it])
    all_images = torch.cat(all_images, dim=0)
    all_voxels = torch.cat(all_voxels, dim=0)
    all_indices = np.concatenate(all_indices, 0)
    print(all_images.shape)
    print(all_voxels.shape)
    np.savez_compressed(f'data/NSD/processed/train_data_{subj}.npz', all_images=all_images, all_voxels=all_voxels, all_indices=all_indices)
        
