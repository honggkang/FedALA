import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('./dataset', dataset, 'train/')

        train_file = train_data_dir + 'train'+ str(idx) + '_.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        # {'x': array([[[[-1., -1., -1., ..., -1., -1., -1.],
        #  [-1., -1., -1., ..., -1., -1.,...1., ..., -1., -1., -1.]]]], dtype=float32), 'y': array([1, 5, 8, ..., 8, 1, 1])}

        return train_data

    else:
        test_data_dir = os.path.join('./dataset', dataset, 'test/')

        test_file = test_data_dir + 'test'+str(idx) + '_.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_data_gefl(dataset, idx, dict_users, is_train=True):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            # transforms.Normalize((0.1307,), (0.3081,)), # DCGAN 0.5
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            # transforms.Normalize((0.1307,), (0.3081,)), # DCGAN 0.5
        ])
    if is_train:
        dataset_train = datasets.MNIST('.data/mnist', train=True, download=True, transform=transform_train)
        train_data = DatasetSplit(dataset_train, dict_users[idx])
        # # {'x': array([[[[-1., -1., -1., ..., -1., -1., -1.],
        # #  [-1., -1., -1., ..., -1., -1.,...1., ..., -1., -1., -1.]]]], dtype=float32), 'y': array([1, 5, 8, ..., 8, 1, 1])}

        return train_data

    else:
        dataset_test = datasets.MNIST('.data/mnist', train=False, download=True, transform=transform_test)
        test_data = dataset_test

        return test_data
    

def read_client_data_gefl(dataset, idx, dict_users, is_train=True):
    if is_train:
        train_data = read_data_gefl(dataset, idx, dict_users, is_train)
        # X_train = torch.Tensor(train_data['x']).type(torch.float32)
        # y_train = torch.Tensor(train_data['y']).type(torch.int64)

        # train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data_gefl(dataset, idx, dict_users, is_train)
        # X_test = torch.Tensor(test_data['x']).type(torch.float32)
        # y_test = torch.Tensor(test_data['y']).type(torch.int64)
        # test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    

def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

