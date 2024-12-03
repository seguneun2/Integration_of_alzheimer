import torch
import pathlib
import json
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import os
from torch.utils.data import Dataset
from torch.nn import functional as F
from config import *


class MRIwithGeneDataset(Dataset):
    #""" _ : APOE4 제거"""
    def __init__(self, mode, seed=47, label_list=LABEL_LIST):

        self.mode = mode
        self.label_list = label_list
        self.data_list = self._load_data_list(mode, seed)
        self.loaded_data_list = self._load_all_data(self.data_list)
        self.label_list = LABEL_LIST

    def _load_data_list(self, mode, seed):
        """        
        Arguments
        ---------
        - mode: 'train' or 'val' or 'test'
        - seed: 랜덤 시드값
        """

        # 라벨 불러오기
        label_df = pd.read_csv(ANNOT_PATH)[["LABEL", "PTID", "PTGENDER"]]
        gene_df = pd.read_csv(GENE_PATH)

        merged_df = pd.merge(label_df, gene_df, how="inner", left_on="PTID", right_on="Unnamed: 0")

        label_dict = {
            merged_df["PTID"][i]: merged_df["LABEL_x"][i]
            for i in range(len(merged_df))
        }

        data_list = []
     
        # 데이터 리스트 로딩
        for i in range(len(merged_df)):
            ptid = merged_df["PTID"][i]
            ptgd = merged_df["PTGENDER"][i]

            path = os.path.join(PREPROCESSED_DATA_PATH, f"{ptid}.nii.gz")
            label = self.label_list.index(label_dict[ptid])
            gene = merged_df[[col for col in merged_df.columns if col.startswith(("rs","APOE"))]].iloc[i].values
            gene[gene == -1] = 3

            data_list.append((path, label, gene, ptgd))
        

        N = len(data_list)

        # Train - val - test split
        if mode == "train":
            start_index = 0
            end_index = int(N * 0.7)
        elif mode == "val":
            start_index = int(N * 0.7)
            end_index = int(N * 0.85)
        elif mode == "test":
            start_index = int(N * 0.85)
            end_index = N
        elif mode == "all":
            start_index = 0
            end_index = N
        else:
            raise KeyError("Dataset mode must be either 'train' or 'test'.")

        # 데이터셋 섞어서 train / val / test split
        np.random.seed(seed)
        np.random.shuffle(data_list)

        data_list = data_list[start_index:end_index]
   
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_one_image(self, path):

        image = nib.load(path).get_fdata()
        image, mask = self._preprocess(image)


        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return image, mask
        
    def _preprocess(self, image3d, eps=1e-12):

        image3d = (image3d - image3d.min(axis=(1, 2), keepdims=True)) / (image3d.max(axis=(1, 2), keepdims=True) - image3d.min(axis=(1, 2), keepdims=True) + 1e-12)
        mask = (image3d > 1e-1).astype(np.float32)

        image3d[mask == 0] = 0
        

        w, h, d= image3d.shape

        image3d = image3d[...,d//10:-d//10] 
        mask = mask[...,d//10:-d//10]

        return image3d, mask

    def _load_one_data(self, path, label, gene, _):
        image3d, mask = self._load_one_image(path)
        return image3d, mask, label, gene

    def _load_all_data(self, data_list):
        print("Dataset are loading...")
        loaded_data_list = list(map(lambda item: self._load_one_data(*item), data_list))
        print("Dataset was loaded.")
        return loaded_data_list

    def _color_distortion(self, image3d):
        image3d = image3d + np.random.rand() * 0.4 - 0.2
        image3d[image3d < 0] = 0
        image3d[image3d > 1] = 1
        return image3d

    def _random_noise(self, image3d):
        scale = np.abs(np.random.normal(loc=0, scale=0.005, size=image3d.shape)) + 1e-6
        image3d = np.random.normal(loc=image3d, scale=scale)
        image3d[image3d < 0] = 0
        image3d[image3d > 1] = 1
        return image3d

    def __getitem__(self, index):
        image3d, mask, label, gene = self.loaded_data_list[index]

        if self.mode == "train" and np.random.rand() < 0.5:
            image3d = self._color_distortion(image3d)

        if self.mode == "train" and np.random.rand() < 0.5:
            image3d = self._random_noise(image3d)
            
        image3d[mask == 0] = 0

        image3d_tensor = torch.tensor(image3d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        image3d_tensor = F.interpolate(image3d_tensor, size=(VOLUME, VOLUME, VOLUME)).squeeze(0)
        
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(VOLUME, VOLUME, VOLUME)).squeeze(0)

        gene = torch.tensor(gene, dtype=torch.long)

        return image3d_tensor, mask_tensor, label, gene
