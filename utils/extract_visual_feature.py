import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class VGGNet(nn.Module):

    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgDataset(Dataset):

    def __init__(self, img, transform):
        self.img = img
        self.transform = transform

    def __getitem__(self, index):
        img = self.img[index].transpose((0, 2, 1))
        img = torch.from_numpy(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img)


# build image transform
scale = transforms.Lambda(lambda x: x / 255.)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transfrom2 = transforms.Compose([scale, normalize])

# load data
mat = h5py.File('/data1/zl/NUS-WIDE-TC21/nus-wide-tc21-iall.mat')
img = mat['IAll']
dataset = ImgDataset(img, transfrom2)
dataloader = DataLoader(dataset, shuffle=False, batch_size=100)

# build model
model = VGGNet().cuda()

# process
pbar = tqdm(dataloader, total=len(dataloader))
feature = []

for x in pbar:
    x = x.cuda()

    with torch.no_grad():
        f = model(x)
        feature.append(f.cpu().numpy())

feature = np.concatenate(feature, axis=0)
print('img num:', feature.shape[0], 'feature size:', feature.shape[-1])
save_file = '/data1/zl/NUS-WIDE-TC21/feature.npy'
np.save(save_file, feature)
