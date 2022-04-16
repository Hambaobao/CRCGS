import h5py
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
from torch.utils.data import DataLoader


class CustomDataSet(Dataset):

    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count


class CustomDataSetWithTrans(Dataset):

    def __init__(self, origin, images, texts, labels, transform):
        self.origin = origin
        self.images = images
        self.texts = texts
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]

        origin = self.origin[index]
        origin = origin.transpose((0, 2, 1))
        origin = self.transform(origin)
        return img, origin, text, label

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count


class CorrelationLabelDataset:
    '''
    Correlation Dataset used for Label based neighbor
    '''

    def __init__(self, images, texts, neb, batch_size=100, gen_round=100):
        self.images = images
        self.texts = texts
        self.neb = neb
        self.batch_size = batch_size
        self.neb_num = neb.shape[-1]
        self.round = gen_round
        self.data_num = len(self.images)

    def __len__(self):
        return self.round

    def __iter__(self):
        sample_num = self.batch_size // 2

        for _ in range(self.round):
            sample_ind = np.random.randint(0, self.data_num, sample_num)
            img_feature = self.images[sample_ind]
            pos_neb_ind = []
            for i in sample_ind:
                ram = np.random.randint(0, self.neb_num)
                while self.neb[i][ram] == -1:  # remove padding
                    ram = np.random.randint(0, self.neb_num)
                pos_neb_ind.append(self.neb[i][ram])

            pos_neb_ind = np.array(pos_neb_ind)
            pos_text_feature = self.texts[pos_neb_ind]

            neg_neb_ind = []
            for i in sample_ind:
                ram = np.random.randint(0, self.data_num)
                while (self.neb[i] == ram).sum() > 0:  # remove pos
                    ram = np.random.randint(0, self.data_num)
                neg_neb_ind.append(ram)
            neg_neb_ind = np.array(neg_neb_ind)
            neg_text_feature = self.texts[neg_neb_ind]

            img_feature = np.concatenate([img_feature, img_feature], axis=0)
            text_feature = np.concatenate([pos_text_feature, neg_text_feature], axis=0)
            label = np.array([1] * sample_num + [0] * sample_num)

            yield torch.from_numpy(img_feature), torch.from_numpy(text_feature), torch.from_numpy(label).long()


class CorrelationFeatureDataset:
    '''
    Correlation Dataset used for Feature based neighbor
    '''

    def __init__(self, images, texts, images_neb, texts_neb, batch_size=100, gen_round=100):
        self.images = images
        self.texts = texts
        self.images_neb = images_neb
        self.texts_neb = texts_neb
        self.batch_size = batch_size
        self.images_neb_num = images_neb.shape[-1]
        self.texts_neb_num = texts_neb.shape[-1]
        self.round = gen_round
        self.data_num = len(self.images)

    def __len__(self):
        return self.round

    def __iter__(self):
        sample_num = self.batch_size // 2

        for _ in range(self.round):
            sample_ind = np.random.randint(0, self.data_num, sample_num)
            pos_sample_ind = sample_ind[:sample_num]

            # sample pos images
            pos_imgs_ind = []
            for i in pos_sample_ind:
                ram = np.random.randint(0, self.images_neb_num)
                while self.images_neb[i][ram] == -1:  # remove padding
                    ram = np.random.randint(0, self.images_neb_num)
                pos_imgs_ind.append(self.images_neb[i][ram])

            # sample pos texts
            pos_text_ind = []
            for i in pos_sample_ind:
                ram = np.random.randint(0, self.texts_neb_num)
                while self.texts_neb[i][ram] == -1:  # remove padding
                    ram = np.random.randint(0, self.texts_neb_num)
                pos_text_ind.append(self.texts_neb[i][ram])

            pos_imgs_ind = np.array(pos_imgs_ind)
            pos_text_ind = np.array(pos_text_ind)
            pos_imgs_feature = self.images[pos_imgs_ind]
            pos_text_feature = self.texts[pos_text_ind]

            # combine positive set
            related_list = set()
            for i in pos_sample_ind:
                for x in self.texts_neb[i]:
                    related_list.add(x)
                for x in self.images_neb[i]:
                    related_list.add(x)
            cover_ind = np.concatenate([np.arange(self.data_num), np.array(list(related_list))], axis=0)
            remove_repeat = np.unique(cover_ind)

            # sample neg text
            sample_ind = np.random.randint(0, len(remove_repeat), sample_num)
            neg_sample_ind = remove_repeat[sample_ind]
            neg_text_feature = self.texts[neg_sample_ind]

            img_feature = np.concatenate([pos_imgs_feature, pos_imgs_feature], axis=0)
            text_feature = np.concatenate([pos_text_feature, neg_text_feature], axis=0)
            label = np.array([1] * sample_num + [0] * sample_num)

            # print(img_feature, text_feature, label)
            yield torch.from_numpy(img_feature).float(), torch.from_numpy(text_feature).float(), torch.from_numpy(label).long()


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def get_loader(path, batch_size):
    img_train = loadmat(path + '/train_img.mat')['train_img']
    img_test = loadmat(path + '/test_img.mat')['test_img']
    img_eval = img_test
    text_train = loadmat(path + '/train_txt.mat')['train_txt']
    text_test = loadmat(path + '/test_txt.mat')['test_txt']
    text_eval = text_test
    label_train = loadmat(path + '/train_img_lab.mat')['train_img_lab']
    label_test = loadmat(path + '/test_img_lab.mat')['test_img_lab']
    label_eval = label_test

    label_train = ind2vec(label_train).astype(int)
    label_test = ind2vec(label_test).astype(int)
    label_eval = label_test

    imgs = {'train': img_train, 'eval': img_eval, 'test': img_test}
    texts = {'train': text_train, 'eval': text_eval, 'test': text_test}
    labels = {'train': label_train, 'eval': label_eval, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x]) for x in ['train', 'eval', 'test']}

    shuffle = {'train': False, 'eval': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'eval', 'test']}

    img_dim = img_train.shape[1]
    txt_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    data_parameter = {}
    data_parameter['img_dim'] = img_dim
    data_parameter['txt_dim'] = txt_dim
    data_parameter['num_class'] = num_class
    return dataloader, data_parameter


def get_correlation_dataloader(path, batch_size, gen_round=100, dataset_type='feature'):
    '''
    Design for correlation model training
    dataset_type argument is used to check the dataset type ('feature', 'label')

    imgs: feature.npy
    text: bow.npy
    '''
    imgs = {'train': None, 'val': None}
    text = {'train': None, 'val': None}

    for key in imgs:
        imgs_data = np.load(path + '/{}_feature.npy'.format(key))
        text_data = np.load(path + '/{}_bow.npy'.format(key))

        imgs[key] = imgs_data
        text[key] = text_data

    if dataset_type == 'feature':
        nebs = {'train': [None, None], 'val': [None, None]}

        for key in nebs:
            imgs_neb = np.load(path + '/{}_imgs_neb.npy'.format(key))
            text_neb = np.load(path + '/{}_text_neb.npy'.format(key))

            nebs[key][0] = imgs_neb
            nebs[key][1] = text_neb

        dataloader = {x: CorrelationFeatureDataset(imgs[x], text[x], nebs[x][0], nebs[x][1], batch_size=batch_size, gen_round=gen_round) for x in ['train', 'val']}
    else:
        nebs = {'train': None, 'val': None}

        for key in nebs:
            neb = np.load(path + '/{}_neb.npy'.format(key))
            nebs[key] = neb

        dataloader = {x: CorrelationLabelDataset(imgs[x], text[x], nebs[x], batch_size=batch_size, gen_round=gen_round) for x in ['train', 'val']}

    return dataloader


def get_coco_loader(path, batch_size, from_old=True):
    img = np.load(path + '/feature.npy')
    text = np.load(path + '/bow.npy')
    label = np.load(path + '/label.npy')
    label = label[:, :-1]  # remove the redundant class dims

    # sample same number data
    if from_old:
        train_data_list = np.load(path + '/train_data_list.npy')
        test_data_list = np.load(path + '/test_data_list.npy')
        val_data_list = np.load(path + '/val_data_list.npy')
    else:
        data_list = np.arange(len(img))
        np.random.shuffle(data_list)
        train_data_list = data_list[:10000]
        test_data_list = data_list[10000:15000]
        val_data_list = data_list[15000:20000]
        np.save(path + '/train_data_list.npy', train_data_list)
        np.save(path + '/test_data_list.npy', test_data_list)
        np.save(path + '/val_data_list.npy', val_data_list)

    img_train = img[train_data_list]
    img_eval = img[val_data_list]
    img_test = img[test_data_list]
    text_train = text[train_data_list]
    text_eval = text[val_data_list]
    text_test = text[test_data_list]
    label_train = label[train_data_list]
    label_eval = label[val_data_list]
    label_test = label[test_data_list]

    imgs = {'train': img_train, 'eval': img_eval, 'test': img_test}
    texts = {'train': text_train, 'eval': text_eval, 'test': text_test}
    labels = {'train': label_train, 'eval': label_eval, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x]) for x in ['train', 'eval', 'test']}

    shuffle = {'train': True, 'eval': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'eval', 'test']}

    img_dim = img_train.shape[1]
    txt_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    data_parameter = {}
    data_parameter['img_dim'] = img_dim
    data_parameter['txt_dim'] = txt_dim
    data_parameter['num_class'] = num_class
    return dataloader, data_parameter


def get_flickr_loader(path, batch_size, from_old=True):
    img = np.load(path + '/vgg_19_mirflickr_features.npy')
    text = np.load(path + '/bow.npy')
    label = np.load(path + '/labels.npy')
    # label = label[:, :-1] # remove the redundant class dims
    label = label.astype(np.int)  # convert bool type to int

    # sample same number data
    if from_old:
        train_data_list = np.load(path + '/train_data_list.npy')
        test_data_list = np.load(path + '/test_data_list.npy')
        val_data_list = np.load(path + '/val_data_list.npy')
    else:
        data_list = np.arange(len(img))
        np.random.shuffle(data_list)
        train_data_list = data_list[:10000]
        test_data_list = data_list[10000:12000]
        val_data_list = data_list[12000:14000]
        np.save(path + '/train_data_list.npy', train_data_list)
        np.save(path + '/test_data_list.npy', test_data_list)
        np.save(path + '/val_data_list.npy', val_data_list)

    img_train = img[train_data_list]
    img_eval = img[val_data_list]
    img_test = img[test_data_list]
    text_train = text[train_data_list]
    text_eval = text[val_data_list]
    text_test = text[test_data_list]
    label_train = label[train_data_list]
    label_eval = label[val_data_list]
    label_test = label[test_data_list]

    imgs = {'train': img_train, 'eval': img_eval, 'test': img_test}
    texts = {'train': text_train, 'eval': text_eval, 'test': text_test}
    labels = {'train': label_train, 'eval': label_eval, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x]) for x in ['train', 'eval', 'test']}

    shuffle = {'train': True, 'eval': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'eval', 'test']}

    img_dim = img_train.shape[1]
    txt_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    data_parameter = {}
    data_parameter['img_dim'] = img_dim
    data_parameter['txt_dim'] = txt_dim
    data_parameter['num_class'] = num_class
    return dataloader, data_parameter


def get_flickr_vgg_loader(path, batch_size, pre_transform, from_old=True):
    img = np.load(path + '/feature.npy')
    text = np.load(path + '/bow.npy')
    label = np.load(path + '/label.npy')
    mat = h5py.File(path + '/mirflickr25k-iall.mat', mode='r')
    origin = np.array(mat['IAll'])
    label = label.astype(np.int)  # convert bool type to int

    # sample same number data
    if from_old:
        train_data_list = np.load(path + '/train_data_list.npy')
        test_data_list = np.load(path + '/test_data_list.npy')
        val_data_list = np.load(path + '/val_data_list.npy')
    else:
        data_list = np.arange(len(img))
        np.random.shuffle(data_list)
        train_data_list = data_list[:10000]
        test_data_list = data_list[10000:12000]
        val_data_list = data_list[12000:14000]
        np.save(path + '/train_data_list.npy', train_data_list)
        np.save(path + '/test_data_list.npy', test_data_list)
        np.save(path + '/val_data_list.npy', val_data_list)

    img_train = img[train_data_list]
    img_eval = img[val_data_list]
    img_test = img[test_data_list]
    text_train = text[train_data_list]
    text_eval = text[val_data_list]
    text_test = text[test_data_list]
    label_train = label[train_data_list]
    label_eval = label[val_data_list]
    label_test = label[test_data_list]
    origin_train = origin[train_data_list]
    origin_eval = origin[val_data_list]
    origin_test = origin[test_data_list]

    imgs = {'train': img_train, 'eval': img_eval, 'test': img_test}
    texts = {'train': text_train, 'eval': text_eval, 'test': text_test}
    labels = {'train': label_train, 'eval': label_eval, 'test': label_test}
    origins = {'train': origin_train, 'eval': origin_eval, 'test': origin_test}
    dataset = {x: CustomDataSetWithTrans(origin=origins[x], images=imgs[x], texts=texts[x], labels=labels[x], transform=pre_transform) for x in ['train', 'eval', 'test']}

    shuffle = {'train': True, 'eval': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'eval', 'test']}

    img_dim = img_train.shape[1]
    txt_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    data_parameter = {}
    data_parameter['img_dim'] = img_dim
    data_parameter['txt_dim'] = txt_dim
    data_parameter['num_class'] = num_class
    return dataloader, data_parameter


def get_nus_vgg_loader(path, batch_size, pre_transform, from_old=True):
    img = np.load(path + '/feature.npy')
    text = np.load(path + '/bow.npy')
    label = np.load(path + '/label.npy')

    label = label.astype(np.int)  # convert bool type to int

    # sample same number data
    if from_old:
        train_data_list = np.load(path + '/train_data_list.npy')
        test_data_list = np.load(path + '/test_data_list.npy')
        val_data_list = np.load(path + '/val_data_list.npy')

    img_train = img[train_data_list]
    img_eval = img[val_data_list]
    img_test = img[test_data_list]
    text_train = text[train_data_list]
    text_eval = text[val_data_list]
    text_test = text[test_data_list]
    label_train = label[train_data_list]
    label_eval = label[val_data_list]
    label_test = label[test_data_list]
    origin_train = np.load(path + '/train_origin.npy')
    origin_eval = np.load(path + '/val_origin.npy')
    origin_test = np.load(path + '/test_origin.npy')

    imgs = {'train': img_train, 'eval': img_eval, 'test': img_test}
    texts = {'train': text_train, 'eval': text_eval, 'test': text_test}
    labels = {'train': label_train, 'eval': label_eval, 'test': label_test}
    origins = {'train': origin_train, 'eval': origin_eval, 'test': origin_test}
    dataset = {x: CustomDataSetWithTrans(origin=origins[x], images=imgs[x], texts=texts[x], labels=labels[x], transform=pre_transform) for x in ['train', 'eval', 'test']}

    shuffle = {'train': True, 'eval': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'eval', 'test']}

    img_dim = img_train.shape[1]
    txt_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    data_parameter = {}
    data_parameter['img_dim'] = img_dim
    data_parameter['txt_dim'] = txt_dim
    data_parameter['num_class'] = num_class
    return dataloader, data_parameter