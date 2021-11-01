import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import pdb

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root, mode, hemi):
    # assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_data = np.load('./dataset/parcellation/M-CRIB-S_train_TEA.npy')
        for it in range(len(train_data)):
            item = os.path.join(root, train_data[it][0] + '_rh.pt')
            items.append(item)
        for it in range(len(train_data)):
            item = os.path.join(root, train_data[it][0] + '_lh.pt')
            items.append(item)
        # pdb.set_trace()

    elif mode == 'val':
        val_data = np.load('./dataset/parcellation/M-CRIB-S_val_TEA.npy')

        for it in range(len(val_data)):
            item = os.path.join(root, val_data[it][0] + '_rh.pt')
            items.append(item)
        for it in range(len(val_data)):
            item = os.path.join(root, val_data[it][0] + '_lh.pt')
            items.append(item)

    elif mode == 'test':
        test_data = np.load('./dataset/parcellation/M-CRIB-S_test_TEA.npy')
        for it in range(len(test_data)):
            item = os.path.join(root, test_data[it][0] + '_rh.pt')
            items.append(item)
        for it in range(len(test_data)):
            item = os.path.join(root, test_data[it][0] + '_lh.pt')
            items.append(item)

    else:
        files_list = os.listdir(os.path.join(root, mode))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, mode), files_list[it])
            items.append(item)

    return items


class GeometricDataset(Dataset):

    def __init__(self, mode, root_dir, hemi):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.hemi = hemi
        self.root_dir = root_dir
        self.mode = mode
        self.files = make_dataset(root_dir, mode, hemi)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        data = torch.load(files_path)
        return data
