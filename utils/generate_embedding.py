import torch
import numpy as np

from tqdm import tqdm


def generate_emb(model, dataloader, sample_num=None):
    '''
    Get generated embedding
    '''

    model.eval()
    t_imgs, t_txts, t_labels = [], [], []
    pbar = tqdm(dataloader, total=len(dataloader))

    with torch.no_grad():
        for imgs, txts, labels in pbar:
            if torch.cuda.is_available():
                imgs = imgs.float().cuda()
                txts = txts.float().cuda()

            t_view1_feature, t_view2_feature, *_ = model(imgs, txts)
            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels)

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    return t_imgs, t_txts, t_labels


def generate_binary_emb(model, binary_layer, dataloader, sample_num=None):
    '''
    Get generated binary embedding
    '''

    model.eval()
    binary_layer.eval()

    t_imgs, t_txts, t_labels = [], [], []
    pbar = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for imgs, txts, labels in pbar:
            if torch.cuda.is_available():
                imgs = imgs.float().cuda()
                txts = txts.float().cuda()

            t_view1_feature, t_view2_feature, *_ = model(imgs, txts)

            t_view1_feature = binary_layer(t_view1_feature).sign()
            t_view2_feature = binary_layer(t_view2_feature).sign()

            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels)

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    return t_imgs, t_txts, t_labels
