from __future__ import division, print_function

import torch
import torchvision

import copy
import time
import numpy as np
from collections import OrderedDict

from tqdm import tqdm

from utils.evaluate import fx_calc_map_label

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())

    # conquer multi-label problem
    Sim = (Sim > 0).float()
    return Sim


def calc_binary_loss(view1_feature, view2_feature, center_feature, view1_hash, view2_hash, center_hash, labels_1, labels_2, alpha, beta, gamma):

    def cos(x, y):
        return x.mm(y.t()) / ((x**2).sum(1, keepdim=True).sqrt().mm((y**2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.

    # center loss
    feature_dist = cos(center_feature, center_feature)
    hash_dist = cos(center_hash, center_hash)
    term1 = torch.pow(feature_dist - hash_dist, 2)
    term1 = term1.sum(-1).mean()

    # # cross loss
    hash11 = cos(view1_hash, view1_hash)
    hash12 = cos(view1_hash, view2_hash)
    hash22 = cos(view2_hash, view2_hash)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()

    term21 = ((1 + torch.exp(hash11)).log() - Sim11 * hash11).mean()
    term22 = ((1 + torch.exp(hash12)).log() - Sim12 * hash12).mean()
    term23 = ((1 + torch.exp(hash22)).log() - Sim22 * hash22).mean()
    term2 = term21 + term22 + term23
    # term2 = 0

    # hash loss
    with torch.no_grad():
        # b_hash1 = torch.sign(view1_hash)
        # b_hash2 = torch.sign(view2_hash)
        b_hashc = torch.sign(center_hash)
    # term3 = F.mse_loss(view1_hash, b_hash1) + F.mse_loss(view2_hash, b_hash2) +\
    #         F.mse_loss(center_hash, b_hashc)

    term3 = torch.pow(center_hash - b_hashc, 2)
    term3 = term3.sum(-1).mean()

    total_loss = term1 * alpha + term2 * beta + term3 * gamma
    return total_loss


def train_model_binary(opt, binary_model, extract_model, data_loader, optimizer, device="cpu", scheduler=None):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(binary_model.state_dict())

    opt.best_acc = 0.0
    opt.best_epoch = 0

    # change feature model to evaluation mode and set the gradient false
    extract_model.eval()
    for p in extract_model.parameters():
        p.requires_grad = False

    # Start training
    for epoch in range(opt.epoch):
        print('Epoch {}/{}'.format(epoch + 1, opt.epoch))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            train_loss = OrderedDict({'gen': 0, 'o': 0, 'c': 0})
            test_loss = OrderedDict({'gen': 0})

            pbar = tqdm(data_loader[phase], total=len(data_loader[phase]))
            for imgs, origins, txts, labels in pbar:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.

                    if torch.cuda.is_available():
                        imgs = origins.cuda().float()
                        txts = txts.cuda().float()
                        labels = labels.cuda()

                    if phase == 'eval':
                        # Test the cross-retrieval model in test set.
                        # Calculate the feature and then claculate the Criterion loss.

                        # Please take care that this loss only include the discrimination loss but
                        # without teacher loss.

                        # set model state
                        binary_model.eval()

                        # Forward
                        with torch.no_grad():
                            view1_feature, view2_feature, view1_predict, view2_predict, *_ = extract_model(imgs, txts)
                            center_feature = (view1_feature + view2_feature) / 2.

                        view1_hash = binary_model(view1_feature)
                        view2_hash = binary_model(view2_feature)
                        center_hash = binary_model(center_feature)

                        loss = calc_binary_loss(view1_feature, view2_feature, center_feature, view1_hash, view2_hash, center_hash, labels, labels, opt.alpha, opt.beta, opt.gamma)

                        # statistics
                        test_loss['gen'] += loss.item()
                    else:
                        # Train the binary transformer model

                        binary_model.train()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        with torch.no_grad():
                            view1_feature, view2_feature, view1_predict, view2_predict, stu_img_feats, stu_txt_feats = extract_model(imgs, txts)
                            center_feature = (view1_feature + view2_feature) / 2.

                        view1_hash = binary_model(view1_feature)
                        view2_hash = binary_model(view2_feature)
                        center_hash = binary_model(center_feature)

                        loss_o = calc_binary_loss(view1_feature, view2_feature, center_feature, view1_hash, view2_hash, center_hash, labels, labels, opt.alpha, opt.beta, opt.gamma)

                        loss = loss_o

                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                        # statistics
                        train_loss['gen'] += loss.item()
                        train_loss['o'] += loss_o.item()
                        train_loss['c'] += loss_c.item()

            # report train or evaluate retults
            report = train_loss if phase == 'train' else test_loss
            epoch_loss = report['gen'] / len(data_loader[phase])
            for key, value in report.items():
                print('  {}:{:.2f}'.format(key, value / len(data_loader[phase])), end=',')
            print()

            # Evaluation in the according dataset
            img2txt, txt2img = eval_model_binary_map(extract_model, binary_model, data_loader[phase], sample_num=1000)
            print('{} loss: {:.4f}, {} dataset eval result - img2txt: {:.4f}  txt2img: {:.4f}'.format(phase, epoch_loss, phase, img2txt, txt2img))

            # deep copy the model
            if phase == 'eval':
                test_img_acc_history.append(img2txt)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)
                # save best record
                if (img2txt + txt2img) / 2. > opt.best_acc:
                    opt.best_acc = (img2txt + txt2img) / 2.
                    opt.best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(binary_model.state_dict())

                # lr scheduler update after testing
                if scheduler is not None:
                    scheduler.step((img2txt + txt2img) / 2.)

        print('>>> current best accuracy: {:.6f}, best epoch: {}\n'.format(opt.best_acc, opt.best_epoch))

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best average accuracy: {:4f}'.format(opt.best_acc))
    print('best epoch: {}'.format(opt.best_epoch))

    # load best model weights
    binary_model.load_state_dict(best_model_wts)

    return binary_model


def eval_model_binary_map(extract_model, binary_layer, dataloader, sample_num=None):
    # eval retrieval score in train and val set
    extract_model.eval()
    binary_layer.eval()

    t_imgs, t_txts, t_labels = [], [], []

    with torch.no_grad():
        for imgs, origins, txts, labels in dataloader:
            if torch.cuda.is_available():
                imgs = origins.cuda().float()
                txts = txts.cuda().float()
                labels = labels.cuda()
            t_view1_feature, t_view2_feature, *_ = extract_model(imgs, txts)

            # convert to binary code
            t_view1_feature = binary_layer(t_view1_feature).sign()
            t_view2_feature = binary_layer(t_view2_feature).sign()

            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)

    eval_i2t = fx_calc_map_label(t_imgs, t_txts, t_labels, sample_num=sample_num)
    eval_t2i = fx_calc_map_label(t_txts, t_imgs, t_labels, sample_num=sample_num)

    return eval_i2t, eval_t2i


def train_model_binary_coco(opt, binary_model, extract_model, data_loader, optimizer, device="cpu", scheduler=None):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(binary_model.state_dict())

    opt.best_acc = 0.0
    opt.best_epoch = 0

    # change feature model to evaluation mode and set the gradient false
    extract_model.eval()
    for p in extract_model.parameters():
        p.requires_grad = False

    # Start training
    for epoch in range(opt.epoch):
        print('Epoch {}/{}'.format(epoch + 1, opt.epoch))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            train_loss = OrderedDict({'gen': 0, 'o': 0})
            test_loss = OrderedDict({'gen': 0})

            pbar = tqdm(data_loader[phase], total=len(data_loader[phase]))
            for imgs, txts, labels in pbar:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.

                    if torch.cuda.is_available():
                        imgs = imgs.cuda().float()
                        txts = txts.cuda().float()
                        labels = labels.cuda()

                    if phase == 'eval':
                        # Test the cross-retrieval model in test set.
                        # Calculate the feature and then claculate the Criterion loss.

                        # Please take care that this loss only include the discrimination loss but
                        # without teacher loss.

                        # set model state
                        binary_model.eval()

                        # Forward
                        with torch.no_grad():
                            view1_feature, view2_feature, view1_predict, view2_predict, *_ = extract_model(imgs, txts)
                            center_feature = (view1_feature + view2_feature) / 2.

                        view1_hash = binary_model(view1_feature)
                        view2_hash = binary_model(view2_feature)
                        center_hash = binary_model(center_feature)

                        loss = calc_binary_loss(view1_feature, view2_feature, center_feature, view1_hash, view2_hash, center_hash, labels, labels, opt.alpha, opt.beta, opt.gamma)

                        # statistics
                        test_loss['gen'] += loss.item()
                    else:
                        # Train the binary transformer model

                        binary_model.train()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        with torch.no_grad():
                            view1_feature, view2_feature, view1_predict, view2_predict, stu_img_feats, stu_txt_feats = extract_model(imgs, txts)
                            center_feature = (view1_feature + view2_feature) / 2.

                        view1_hash = binary_model(view1_feature)
                        view2_hash = binary_model(view2_feature)
                        center_hash = binary_model(center_feature)

                        loss_o = calc_binary_loss(view1_feature, view2_feature, center_feature, view1_hash, view2_hash, center_hash, labels, labels, opt.alpha, opt.beta, opt.gamma)

                        loss = loss_o

                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                        # statistics
                        train_loss['gen'] += loss.item()
                        train_loss['o'] += loss_o.item()

            # report train or evaluate retults
            report = train_loss if phase == 'train' else test_loss
            epoch_loss = report['gen'] / len(data_loader[phase])
            for key, value in report.items():
                print('  {}:{:.2f}'.format(key, value / len(data_loader[phase])), end=',')
            print()

            # Evaluation in the according dataset
            img2txt, txt2img = eval_model_binary_map_coco(extract_model, binary_model, data_loader[phase], sample_num=1000)
            print('{} loss: {:.4f}, {} dataset eval result - img2txt: {:.4f}  txt2img: {:.4f}'.format(phase, epoch_loss, phase, img2txt, txt2img))

            # deep copy the model
            if phase == 'eval':
                test_img_acc_history.append(img2txt)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)
                # save best record
                if (img2txt + txt2img) / 2. > opt.best_acc:
                    opt.best_acc = (img2txt + txt2img) / 2.
                    opt.best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(binary_model.state_dict())

                # lr scheduler update after testing
                if scheduler is not None:
                    scheduler.step((img2txt + txt2img) / 2.)

        print('>>> current best accuracy: {:.6f}, best epoch: {}\n'.format(opt.best_acc, opt.best_epoch))

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best average accuracy: {:4f}'.format(opt.best_acc))
    print('best epoch: {}'.format(opt.best_epoch))

    # load best model weights
    binary_model.load_state_dict(best_model_wts)

    return binary_model


def eval_model_binary_map_coco(extract_model, binary_layer, dataloader, sample_num=None):
    # eval retrieval score in train and val set
    extract_model.eval()
    binary_layer.eval()

    t_imgs, t_txts, t_labels = [], [], []

    with torch.no_grad():
        for imgs, txts, labels in dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda().float()
                txts = txts.cuda().float()
                labels = labels.cuda()
            t_view1_feature, t_view2_feature, *_ = extract_model(imgs, txts)

            # convert to binary code
            t_view1_feature = binary_layer(t_view1_feature).sign()
            t_view2_feature = binary_layer(t_view2_feature).sign()

            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)

    eval_i2t = fx_calc_map_label(t_imgs, t_txts, t_labels, sample_num=sample_num)
    eval_t2i = fx_calc_map_label(t_txts, t_imgs, t_labels, sample_num=sample_num)

    return eval_i2t, eval_t2i