import numpy as np

from load_data import ind2vec
from scipy.io import loadmat
from tqdm import tqdm


def split_dataset(path, val_rate=0.2, test_rate=0.1):
    imgs = np.load(path + '/feature.npy')
    text = np.load(path + '/bow.npy')
    label = np.load(path + '/label.npy')

    total_num = imgs.shape[0]
    val_num = int(total_num * val_rate)
    test_num = int(total_num * test_rate)
    train_num = total_num - val_num - test_num
    print('Data total num {} -> train:{}, val:{}, test:{}...'.format(total_num, train_num, val_num, test_num))

    data_list = np.arange(len(text))
    np.random.shuffle(data_list)
    val_ind = data_list[:val_num]
    test_ind = data_list[val_num:val_num + test_num]
    train_ind = data_list[val_num + test_num:]

    # save train/val/test feature
    np.save(path + '/train_feature.npy', imgs[train_ind])
    np.save(path + '/train_bow.npy', text[train_ind])
    np.save(path + '/train_label.npy', label[train_ind])

    np.save(path + '/val_feature.npy', imgs[val_ind])
    np.save(path + '/val_bow.npy', text[val_ind])
    np.save(path + '/val_label.npy', label[val_ind])

    np.save(path + '/test_feature.npy', imgs[test_ind])
    np.save(path + '/test_bow.npy', text[test_ind])
    np.save(path + '/test_label.npy', label[test_ind])

    print('Split Finished...')


def get_neb(label, path, neb_num=None, convert_onehot=True):
    '''
    Load data from disk, use label to get similar nebour
    '''
    label_train = label

    train_num = len(label)
    if convert_onehot:
        one_hot_label_train = ind2vec(label_train).astype(int)
    else:
        one_hot_label_train = label_train

    train_neb = np.ones([train_num, neb_num], dtype=np.int)
    train_neb.fill(-1)  # use -1 for padding
    label_sim = one_hot_label_train @ one_hot_label_train.T
    similar_num = (label_sim > 0).sum(-1)
    sort_label_ind = np.argsort(label_sim, axis=-1)

    for i in tqdm(range(train_num)):
        num = neb_num if similar_num[i] >= neb_num else similar_num[i]
        ind = sort_label_ind[i][::-1]
        train_neb[i][:num] = ind[:num]

    np.save('{}/train_neb.npy'.format(path), train_neb)


def get_semantic_neb(imgs, text, imgs_neb, text_neb, path):
    '''
    Load data from disk, use feature to get similar nebour
    '''
    norm_imgs = np.linalg.norm(imgs, axis=-1)
    norm_text = np.linalg.norm(text, axis=-1)
    norm_imgs = imgs / norm_imgs.reshape(-1, 1)
    norm_text = text / norm_text.reshape(-1, 1)
    print(norm_imgs[32], np.linalg.norm(norm_imgs[32]))
    train_num = len(imgs)

    train_imgs_neb = np.ones([train_num, imgs_neb], dtype=np.int)
    train_text_neb = np.ones([train_num, text_neb], dtype=np.int)
    imgs_sim = norm_imgs @ norm_imgs.T
    text_sim = norm_text @ norm_text.T
    sort_imgs_ind = np.argsort(imgs_sim, axis=-1)
    sort_text_ind = np.argsort(text_sim, axis=-1)

    for i in tqdm(range(train_num)):
        ind = sort_imgs_ind[i][::-1]
        train_imgs_neb[i][:imgs_neb] = ind[:imgs_neb]

        ind = sort_text_ind[i][::-1]
        train_text_neb[i][:text_neb] = ind[:text_neb]

    np.save('{}/val_imgs_neb.npy'.format(path), train_imgs_neb)
    np.save('{}/val_text_neb.npy'.format(path), train_text_neb)


def split_flickr_dataset(path):
    data_list = np.load(path + '/data_list.npy')
    np.random.shuffle(data_list)

    train_data_list = data_list[0:10000]
    test_data_list = data_list[10000:12000]
    val_data_list = data_list[12000:14000]

    np.save('{}/train_data_list.npy'.format(path), train_data_list)
    np.save('{}/test_data_list.npy'.format(path), test_data_list)
    np.save('{}/val_data_list.npy'.format(path), val_data_list)


def split_flickr2_dataset(path):
    label = np.load(path + '/label.npy')
    num = len(label)
    data_list = np.arange(num)
    np.random.shuffle(data_list)

    train_data_list = data_list[0:10000]
    test_data_list = data_list[10000:12000]
    val_data_list = data_list[12000:14000]

    np.save('{}/train_data_list.npy'.format(path), train_data_list)
    np.save('{}/test_data_list.npy'.format(path), test_data_list)
    np.save('{}/val_data_list.npy'.format(path), val_data_list)


def get_flickr_semantic_neb(imgs, text, imgs_neb, text_neb, path):
    '''
    Load data from disk, use feature to get similar nebour
    '''
    norm_imgs = np.linalg.norm(imgs, axis=-1)
    norm_text = np.linalg.norm(text, axis=-1)
    norm_imgs = imgs / norm_imgs.reshape(-1, 1)
    norm_text = text / norm_text.reshape(-1, 1)
    print(norm_imgs[32], np.linalg.norm(norm_imgs[32]))
    train_num = len(imgs)

    train_imgs_neb = np.ones([train_num, imgs_neb], dtype=np.int)
    train_text_neb = np.ones([train_num, text_neb], dtype=np.int)
    imgs_sim = norm_imgs @ norm_imgs.T
    text_sim = norm_text @ norm_text.T
    sort_imgs_ind = np.argsort(imgs_sim, axis=-1)
    sort_text_ind = np.argsort(text_sim, axis=-1)

    for i in tqdm(range(train_num)):
        ind = sort_imgs_ind[i][::-1]
        train_imgs_neb[i][:imgs_neb] = ind[:imgs_neb]

        ind = sort_text_ind[i][::-1]
        train_text_neb[i][:text_neb] = ind[:text_neb]

    np.save('{}/val_data_list_imgs_neb2.npy'.format(path), train_imgs_neb)
    np.save('{}/val_data_list_text_neb2.npy'.format(path), train_text_neb)


def split_nus_dataset(path):
    '''
    A split function to enforce randomly select 100 images
    per class as the query set, and 500 images per class as the training set.
    '''

    label = np.load(path + '/label.npy')
    text = np.load(path + '/bow.npy')
    num = len(label)
    data_list = np.arange(num)
    # since there are some images don't have bow
    data_list = data_list[np.sum(text[data_list], axis=1) > 0]
    print('Valid list: {}'.format(len(data_list)))
    np.save('{}/legal_data_list.npy'.format(path), data_list)

    def sample(source_list, sample_num):
        # sample_data_list = np.random.choice(source_list, sample_num, replace=False)
        # while np.sum(np.sum(label[sample_data_list], axis=0) >= sample_num // class_num) != class_num:
        #     print(np.sum(label[sample_data_list], axis=0))
        #     sample_data_list = np.random.choice(source_list, sample_num, replace=False)

        label_dist = np.sum(label[source_list], axis=0)
        sample_order = np.argsort(label_dist)
        class_num = label.shape[-1]

        sample_data_list = []
        for order in sample_order:
            filter_list = source_list[label[source_list][:, order] == 1]
            sample_list = np.random.choice(filter_list, sample_num // class_num, replace=False)
            sample_data_list.append(sample_list)
            source_list = np.setdiff1d(source_list, sample_list)

        sample_data_list = np.concatenate(sample_data_list)
        # check_unique
        assert len(np.unique(sample_data_list)) == sample_num
        # check balance
        assert np.all(np.sum(label[sample_data_list], axis=0) >= sample_num // class_num)

        return sample_data_list

    # sample test set
    test_data_list = sample(data_list, 21 * 100)
    data_list = np.setdiff1d(data_list, test_data_list)

    # sample val set
    val_data_list = sample(data_list, 21 * 100)
    data_list = np.setdiff1d(data_list, val_data_list)

    # sample train set
    train_data_list = sample(data_list, 21 * 500)
    data_list = np.setdiff1d(data_list, train_data_list)

    # print balance label
    print('Train: {}'.format(len(train_data_list)), np.sum(label[train_data_list], axis=0))
    print('Val: {}'.format(len(val_data_list)), np.sum(label[val_data_list], axis=0))
    print('Test: {}'.format(len(test_data_list)), np.sum(label[test_data_list], axis=0))

    # assert not convered
    assert len(np.union1d(np.union1d(train_data_list, val_data_list), test_data_list)) == (2100 * 2 + 21 * 500)

    np.save('{}/train_data_list.npy'.format(path), train_data_list)
    np.save('{}/test_data_list.npy'.format(path), test_data_list)
    np.save('{}/val_data_list.npy'.format(path), val_data_list)


def get_nus_semantic_neb(imgs, text, imgs_neb, text_neb, path):
    '''
    Load data from disk, use feature to get similar nebour
    '''
    norm_imgs = np.linalg.norm(imgs, axis=-1)
    norm_text = np.linalg.norm(text, axis=-1)
    norm_imgs = imgs / norm_imgs.reshape(-1, 1)
    norm_text = text / norm_text.reshape(-1, 1)
    print(norm_imgs[32], np.linalg.norm(norm_imgs[32]))
    train_num = len(imgs)

    train_imgs_neb = np.ones([train_num, imgs_neb], dtype=np.int)
    train_text_neb = np.ones([train_num, text_neb], dtype=np.int)
    imgs_sim = norm_imgs @ norm_imgs.T
    text_sim = norm_text @ norm_text.T
    sort_imgs_ind = np.argsort(imgs_sim, axis=-1)
    sort_text_ind = np.argsort(text_sim, axis=-1)

    for i in tqdm(range(train_num)):
        ind = sort_imgs_ind[i][::-1]
        train_imgs_neb[i][:imgs_neb] = ind[:imgs_neb]

        ind = sort_text_ind[i][::-1]
        train_text_neb[i][:text_neb] = ind[:text_neb]

    np.save('{}/train_data_list_imgs_neb.npy'.format(path), train_imgs_neb)
    np.save('{}/train_data_list_text_neb.npy'.format(path), train_text_neb)


if __name__ == '__main__':
    path = '/data1/zl/NUS-WIDE-TC21/'

    # split_nus_dataset(path)
    # split_dataset(path)
    # label_train = loadmat(path + "/train_img_lab.mat")['train_img_lab']
    # label_train = np.load(path + '/train_label.npy')
    # get_neb(label_train, path, neb_num=30, convert_onehot=False)
    imgs = np.load(path + '/feature.npy')
    text = np.load(path + '/bow.npy')
    data_list = np.load(path + '/train_data_list.npy')
    imgs = imgs[data_list]
    text = text[data_list]
    get_nus_semantic_neb(imgs, text, 10, 6, path)
