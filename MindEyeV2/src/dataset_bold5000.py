from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from collections import defaultdict

from PIL import Image


def identity(x):
    return x

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def pad_to_length_dim2(x, length):
    assert x.ndim == 2
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]
    

def create_BOLD5000_dataset_classify(path='/home/users/ywei13/playground/BrainGen/data/BOLD5000/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False, do_normalize=False,
            target_sub_train_proportion=1, target_sub='CSI1'):
    #通过subjects参数传入目标被试，例如需要CS1的数据，则subjects=['CS1'],需要传入list
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']

    #fmri_path对应文件夹中存放有BOLD5000的四个被试CS1-CS4的fmri文件，已经按照脑区抽取为np array存放。
    #每个文件大小为5254*size(ROI), size(ROI)为对应脑区包含的voxel数量，可理解为特征维度。 5254为被试看到的图片数量
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/')
    # img_path对应文件夹中存放有被试看到的图片
    img_path = os.path.join(path, 'BOLD5000_Stimuli')

    # imgs_dict中存放了被试看到的所有图片的RGB pixel matrix, 尺寸为256*256*3，如需使用需除以255
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'), allow_pickle=True).item()

    # repeated_imgs_list中存放了重复采集的所有图片的文件名，重复采集的图像将作为测试集
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = {}
    img_train_major = {}

    img_name_train_major = {}

    if type(subjects) == str:
        subjects = [subjects]

    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        # fmri_file_name = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
                    # fmri_file_name.append(os.path.join(fmri_path, npy))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        
        if do_normalize:
            fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))

        # load image
        # 图像在实验中呈现给被试者的顺序存放在Stimuli_Presentation_Lists文件夹中
        # img_files， 即get_stimuli_list返回的列表中包含按呈现顺序排列的图像文件名
        img_files = get_stimuli_list(img_path, sub)
        # img_data_sub中存放了image的pixel matrix
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        # test split
        # 找出重复采集的图片在img_files中的index
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        #按index取出fmri，因可能存在重复采集，需要将重复采集的fmri做平均
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        #按index取出图片
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])

        # test_fmri_name = [fmri_file_name[idx[0]] for idx in test_idx]
        # 按index取出图片文件名
        test_img_name = np.stack([img_files[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)
            test_img_name = np.concatenate([test_img_name, np.stack([img_files[idx] for idx in test_idx_flatten])], axis=0)

        # train split
        # 找出测试数据以外其他数据的index
        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        # train_fmri_name = fmri_file_name[train_idx]
        train_img_name = [img_files[idx] for idx in train_idx]

        if target_sub_train_proportion < 1:
            train_fmri = train_fmri[:int(train_fmri.shape[0]*target_sub_train_proportion)]
            train_img  = train_img[:int(train_img.shape[0]*target_sub_train_proportion)]

        fmri_train_major[sub] = train_fmri
        img_train_major[sub] = train_img
        img_name_train_major[sub] = train_img_name
        
        if sub == target_sub:
            fmri_test_major = test_fmri
            img_test_major = test_img
            img_name_test_major = test_img_name
        
    # fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    # fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    # img_train_major = np.concatenate(img_train_major, axis=0)
    # img_test_major = np.concatenate(img_test_major, axis=0)

    # img_name_test_major = np.concatenate(img_name_test_major, axis=0)
    # img_name_train_major = np.concatenate(img_name_train_major, axis=0)
    
    train_sets = []
    for sub in subjects:
        num_voxels = fmri_train_major[sub].shape[-1]
        if isinstance(image_transform, list):
            train_sets.append(BOLD5000_dataset_classify(fmri_train_major[sub], img_train_major[sub], img_name_train_major[sub], fmri_transform, image_transform[0], num_voxels))
            if sub == target_sub:
                test_set = BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform[1], num_voxels)
            # test_sets.append(BOLD5000_dataset_classify(fmri_test_major[sub], img_test_major[sub], img_name_test_major[sub], torch.FloatTensor, image_transform[1], num_voxels))
        else:
            train_sets.append(BOLD5000_dataset_classify(fmri_train_major[sub], img_train_major[sub], img_name_train_major[sub], fmri_transform, image_transform, num_voxels))
            if sub == target_sub:
                test_set = BOLD5000_dataset_classify(fmri_test_major, img_test_major, img_name_test_major, torch.FloatTensor, image_transform, num_voxels)
            # test_sets.append(BOLD5000_dataset_classify(fmri_test_major[sub], img_test_major[sub], img_name_test_major[sub], torch.FloatTensor, image_transform, num_voxels))
    return train_sets, test_set


class BOLD5000_dataset_classify(Dataset):
    def __init__(self, fmri, image, image_name, fmri_transform=identity, image_transform=identity, num_voxels=0):
        self.fmri = torch.tensor(fmri, dtype=torch.float32)
        # self.image = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2)
        self.image = image
        # self.fmri_name = fmri_name
        self.image_name = image_name

        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.imagename2index = defaultdict(list)
        for i, iname in enumerate(self.image_name):
            self.imagename2index[iname].append(i)

        self.resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
            
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index]
        # fmri_name = self.fmri_name[index]
        img_name = self.image_name[index]
        fmri = np.expand_dims(fmri, axis=0) 
        if self.image_transform == identity:
            return {'fmri': self.fmri_transform(fmri), 
                    'image': self.resize(img),
                    'image_name': img_name,
                    'data_index': index}
        else:
            return {'fmri': self.fmri_transform(fmri), 
                    'image': self.image_transform(img),
                    'image_name': img_name,
                    'data_index': index}
    
    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass

if __name__ == '__main__':
    create_BOLD5000_dataset_classify(subjects=['CSI1', 'CSI2', 'CSI3', 'CSI4'])