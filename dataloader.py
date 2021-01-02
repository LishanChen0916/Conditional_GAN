import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class JsonDataloader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        if self.mode == 'train':
            self.img_name, self.conditions = readJson(self.root, self.mode)
        else:
            self.conditions = readJson(self.root, self.mode)
        self.objects_dict = CharDict().word2index
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, index):
        if self.mode == 'train':
            path = os.path.join(self.root + '/iclevr', self.img_name[index])
            # From RGBA to RGB
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            condition = self.conditions[index]
            condition = torch.LongTensor([self.objects_dict[word] for word in condition])
            return img, self.conditionToOnehot(condition)
        else:
            condition = self.conditions[index]
            condition = torch.LongTensor([self.objects_dict[word] for word in condition])
            return self.conditionToOnehot(condition)
    
    def conditionToOnehot(self, condition):
        onehot = np.zeros(24)
        for idx in condition:
            onehot[idx] += 1

        return onehot

class CharDict:
    def __init__(self):
        self.word2index = {}
        self.n_words = 0

        objects_path = 'dataset/objects.json'
        with open(objects_path, 'r') as reader:
            for item in json.load(reader):
                self.addWord(item)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.n_words += 1

def readJson(root, mode):
    path = os.path.join(root, mode + '.json')
    with open(path, 'r') as reader:
        data_ = json.load(reader)
        img_name = []
        conditions = []

        if mode == 'train':
            for item in data_:
                img_name.append(item)
                conditions.append(data_[item])

            return np.array(img_name), np.array(conditions)

        else:
            for item in data_:
                conditions.append(item)
            
            return np.array(conditions)
