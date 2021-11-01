import os
import torch
from torch.utils.data import Dataset


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    # assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train_dm1':
        oth = 'train_dm2'
        files_lab_list = os.listdir(os.path.join(root, 'train_dm1'))
        files_unl_list = os.listdir(os.path.join(root, oth))
        ratio = int(files_unl_list.__len__() / files_lab_list.__len__())

        files_lab_list.sort()
        for it in range(len(files_lab_list)):
            item = os.path.join(os.path.join(root, 'train_dm1'), files_lab_list[it])
            items.append(item)

        items = items * int(ratio)

    elif mode == 'val':
        files_list = os.listdir(os.path.join(root, 'valid'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'valid'), files_list[it])
            items.append(item)

    elif mode == 'test':
        files_list = os.listdir(os.path.join(root, 'test'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'test'), files_list[it])
            items.append(item)

    elif mode == 'both':
        files_list1 = os.listdir(os.path.join(root, 'train_dm1'))
        files_list2 = os.listdir(os.path.join(root, 'train_dm2'))
        files_list = files_list1[:107]
        files_list.extend(files_list2[:107])
        files_list.sort()

        for it in range(len(files_list)):
            if os.path.exists(os.path.join(os.path.join(root, 'train_dm1'), files_list[it])):
                item = os.path.join(os.path.join(root, 'train_dm1'), files_list[it])
            else:
                item = os.path.join(os.path.join(root, 'train_dm2'), files_list[it])
            items.append(item)

    else:
        files_list = os.listdir(os.path.join(root, mode))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, mode), files_list[it])
            items.append(item)

    return items


class GeometricDataset(Dataset):

    def __init__(self, mode, root_dir):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        data = torch.load(files_path)
        return data
