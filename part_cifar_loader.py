from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle



class PART_CIFAR10(Dataset):
    """`HALF_CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    train_list = ['data_batch_1','data_batch_2','data_batch_3',
                  'data_batch_4','data_batch_5']
    test_list = ['test_batch']
    all_classes = [0,1,2,3,4,5,6,7,8,9]
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, prune_classes=all_classes,
                 fine_tune_classes=all_classes,
                 prune_rate=1, fine_tune=False):

        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.root = root
        self.train = train  # training set or test set
        self.fine_tune = fine_tune
        if self.download:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            datasets.CIFAR10(os.path.join(self.root,'origin_data'),\
                             train=True, download=True, transform=transform)
            self.Paser_data(prune_classes, fine_tune_classes, prune_rate)
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        if not self.fine_tune:
            datapath = os.path.join(self.root, 'parsed_data', 'prune_data')
        else:
            datapath = os.path.join(self.root, 'parsed_data','fine_tune_data')
        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(datapath,  file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'] % 5)
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def Paser_data(self,prune_classes, fine_tune_classes, prune_rate):
        base_path = os.path.join(self.root,'origin_data/cifar-10-batches-py')
        total_list = self.train_list + self.test_list
        for item in total_list:
            prune_label = np.array([])
            prune_data = np.array([])
            prune_file_name = np.array([])
            fine_tune_label = np.array([])
            fine_tune_data = np.array([])
            fine_tuen_file_name = np.array([])
            prune_iter = int(1 / prune_rate)
            with open(os.path.join(base_path,item),'rb') as f:
                data = pickle.load(f, encoding='latin1')
            size = len(data['labels'])
            iter_ = 0
            for i in range(size):
                if data['labels'][i] in prune_classes and iter_ % prune_iter == 0:
                    iter_ += 1
                    prune_label = np.append(prune_label,data['labels'][i])
                    prune_data = np.append(prune_data,data['data'][i])
                    prune_file_name = np.append(prune_file_name,data['filenames'][i])
                if data['labels'][i] in fine_tune_classes and data['labels'][i] in prune_classes\
                        and i % prune_iter != 0:
                    iter_ += 1
                    fine_tune_label = np.append(fine_tune_label,data['labels'][i])
                    fine_tune_data = np.append(fine_tune_data,data['data'][i])
                    fine_tuen_file_name = np.append(fine_tuen_file_name,data['filenames'][i])
                if data['labels'][i] in fine_tune_classes and data['labels'][i] not in prune_classes:
                    fine_tune_label = np.append(fine_tune_label,data['labels'][i])
                    fine_tune_data = np.append(fine_tune_data,data['data'][i])
                    fine_tuen_file_name = np.append(fine_tuen_file_name,data['filenames'][i])
            new_dataset = {}
            new_dataset['labels'] = prune_label
            new_dataset['data'] = prune_data
            new_dataset['filenames'] = prune_file_name
            utils.checkdir(os.path.join('parsed_data','prune_data'))
            with open(os.path.join('parsed_data','prune_data',item),'wb') as f:
                pickle.dump(new_dataset, f, 0)
            new_dataset = {}
            new_dataset['labels'] = fine_tune_label
            new_dataset['data'] = fine_tune_data
            new_dataset['filenames'] = fine_tuen_file_name
            utils.checkdir(os.path.join('parsed_data','fine_tune_data'))
            with open(os.path.join('parsed_data','fine_tune_data',item),'wb') as f:
                pickle.dump(new_dataset, f, 0)
        data = {}
        prune_label = np.array([])
        prune_data = np.array([])
        prune_file_name = np.array([])
        fine_tune_label = np.array([])
        fine_tune_data = np.array([])
        fine_tuen_file_name = np.array([])
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.uint8)
        target = target.astype(np.long)
        # print()
        # print('target type:',type(target))
        # print('target',target)
        # print('train:',self.train)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)




class CIFAR100(PART_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


