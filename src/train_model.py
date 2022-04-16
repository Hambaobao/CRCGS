from __future__ import division, print_function

import copy
import time

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torchvision
import torch.nn.functional as F
from torch.nn import DataParallel

from utils.evaluate import fx_calc_map_label
from utils.model_loss import contrastive_loss
from utils.model_loss import support_set_loss as ssl

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())

    # conquer multi-label problem
    Sim = (Sim > 0).float()
    return Sim


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, phi, alpha, beta):
    # term1 is a multi-label classification problem ...
    term1 = F.binary_cross_entropy_with_logits(view1_predict, labels_1.float()) + \
        F.binary_cross_entropy_with_logits(view2_predict, labels_2.float())

    def cos(x, y):
        return x.mm(y.t()) / ((x**2).sum(1, keepdim=True).sqrt().mm((y**2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.

    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = phi * term1 + alpha * term2 + beta * term3
    return im_loss


def train_model(opt, data_loader, model_s, output_device, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt, optimizer, ta_optimizer, scheduler):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_state = copy.deepcopy(model_s.state_dict())
    opt.best_acc = 0.0
    opt.best_epoch = 0

    # change teacher model to evaluation mode and set the gradient false
    teacher_i2t.eval()
    teacher_t2i.eval()
    for p in teacher_i2t.parameters():
        p.requires_grad = False
    for p in teacher_t2i.parameters():
        p.requires_grad = False

    # Start training
    for epoch in range(opt.epoch):
        print('Epoch {}/{}'.format(epoch + 1, opt.epoch))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            train_loss = OrderedDict({'gen': 0., 'o': 0., 't': 0., 'ta': 0., 'dis': 0., 'c': 0., 'ssl': 0.})
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
                        model_s.eval()

                        # Forward
                        view1_feature, view2_feature, view1_predict, view2_predict, *_ = model_s(imgs, txts)

                        loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, opt.phi, opt.alpha, opt.beta)

                        # statistics
                        test_loss['gen'] += loss.item()
                    else:
                        # Train the model in an adversarial mode

                        # Discriminator Forward
                        model_s.eval()
                        tea_ass_img.train()
                        tea_ass_txt.train()
                        for _ in range(opt.dis_iter):
                            # zero the parameter gradients
                            ta_optimizer.zero_grad()

                            # generatarion feature
                            with torch.no_grad():
                                view1_feature, view2_feature, view1_predict, view2_predict, \
                                    stu_img_feats, stu_txt_feats = model_s(imgs, txts)

                                tea_img_feats = teacher_i2t.get_embedding(imgs)
                                tea_txt_feats = teacher_t2i.get_embedding(txts)

                                # adversarial loss
                                img_sample_num = stu_img_feats.shape[0]
                                txt_sample_num = stu_txt_feats.shape[0]
                                img_hidden = torch.cat([stu_img_feats, tea_img_feats], dim=0)
                                txt_hidden = torch.cat([stu_txt_feats, tea_txt_feats], dim=0)
                                img_predict = teacher_t2i.get_generation(txt_hidden)
                                txt_predict = teacher_i2t.get_generation(img_hidden)

                            # get training label
                            img_label = torch.cat([torch.zeros(img_sample_num), torch.ones(img_sample_num)]).long()
                            txt_label = torch.cat([torch.zeros(txt_sample_num), torch.ones(txt_sample_num)]).long()
                            img_label = img_label.to(imgs.device)
                            txt_label = txt_label.to(txts.device)

                            # cal loss
                            d_loss = tea_ass_img.train_loss(img_predict, img_label) + tea_ass_txt.train_loss(txt_predict, txt_label)
                            d_loss.backward()
                            ta_optimizer.step()

                            # statistics
                            train_loss['dis'] += d_loss.item() / opt.dis_iter

                        # Generator Forward
                        model_s.train()
                        tea_ass_img.eval()
                        tea_ass_txt.eval()
                        for _ in range(opt.gen_iter):
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward model
                            view1_feature, view2_feature, view1_predict, view2_predict, \
                                stu_img_feats, stu_txt_feats = model_s(imgs, txts)

                            # get teacher embedding
                            with torch.no_grad():
                                tea_img_feats = teacher_i2t.get_embedding(imgs)
                                tea_txt_feats = teacher_t2i.get_embedding(txts)

                            # old discrimination loss
                            loss_o = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, opt.phi, opt.alpha, opt.beta)

                            # teacher loss
                            loss_t = opt.w_t * (F.mse_loss(stu_img_feats, tea_img_feats) + F.mse_loss(stu_txt_feats, tea_txt_feats))
                            # teaching assistant loss
                            img_predict = teacher_t2i.get_generation(stu_txt_feats)
                            txt_predict = teacher_i2t.get_generation(stu_img_feats)
                            loss_ta = opt.w_ta * (tea_ass_img.adversarial_loss(img_predict) + tea_ass_txt.adversarial_loss(txt_predict))
                            # contrastive loss
                            loss_c = opt.w_c * contrastive_loss(tea_img_feats, tea_txt_feats, stu_img_feats, stu_txt_feats, opt.batch_size, len(data_loader['train']), opt.cll_t) if opt.contrast else torch.tensor(0.)

                            # support set loss
                            loss_ssl = opt.w_ssl * ssl(stu_img_feats, stu_txt_feats, labels, len(data_loader['train']), opt.ssl_t)

                            # backward + optimize only if in training phase
                            loss = loss_o + loss_t + loss_ta + loss_c + loss_ssl

                            loss.backward()
                            optimizer.step()

                            # statistics
                            train_loss['gen'] += loss.item() / opt.gen_iter
                            train_loss['o'] += loss_o.item() / opt.gen_iter
                            train_loss['t'] += loss_t.item() / opt.gen_iter
                            train_loss['ta'] += loss_ta.item() / opt.gen_iter
                            train_loss['c'] += loss_c.item() / opt.gen_iter
                            train_loss['ssl'] += loss_ssl.item() / opt.gen_iter

            # report
            string = []
            epoch_loss = 0
            if phase == 'train':
                report = train_loss
                epoch_loss = train_loss['gen'] / len(data_loader[phase])
            else:
                report = test_loss
                epoch_loss = test_loss['gen'] / len(data_loader[phase])
            for key, val in report.items():
                string.append("{}:{:.2f}".format(key, val / len(data_loader[phase])))
            print(', '.join(string))

            # Evaluation in the according dataset
            eval_i2t, eval_t2i = eval_model_map(model_s, data_loader[phase], sample_num=1000)
            print('{} Loss: {:.4f}, {} dataset eval result - img2txt: {:.4f}  txt2img: {:.4f}'.format(phase, epoch_loss, phase, eval_i2t, eval_t2i))

            # deep copy the model
            if phase == 'eval':
                test_img_acc_history.append(eval_i2t)
                test_txt_acc_history.append(eval_t2i)
                epoch_loss_history.append(epoch_loss)
                # save best record
                if (eval_i2t + eval_t2i) / 2. > opt.best_acc:
                    opt.best_acc = (eval_i2t + eval_t2i) / 2.
                    opt.best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(get_state_dict(model_s))
                # lr scheduler update after evaluate
                if scheduler is not None:
                    scheduler.step((eval_i2t + eval_t2i) / 2.)
        print('>>> current best accuracy: {:.6f}, best epoch: {}\n'.format(opt.best_acc, opt.best_epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(opt.best_acc))
    print('best epoch: {}'.format(opt.best_epoch))

    # load best model weights
    if isinstance(model_s, DataParallel):
        model_s.module.load_state_dict(best_model_state)
    else:
        model_s.load_state_dict(best_model_state)

    return test_img_acc_history, test_txt_acc_history, epoch_loss_history


def eval_model_map(model, dataloader, sample_num=None):
    # eval retrieval score in train and val set
    model.eval()
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for imgs, txts, labels in dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda().float()
                txts = txts.cuda().float()
                labels = labels.cuda()
            t_view1_feature, t_view2_feature, *_ = model(imgs, txts)
            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    eval_i2t = fx_calc_map_label(t_imgs, t_txts, t_labels, sample_num=sample_num)
    eval_t2i = fx_calc_map_label(t_txts, t_imgs, t_labels, sample_num=sample_num)
    return eval_i2t, eval_t2i


def train_model_with_vgg(opt, data_loader, model_s, output_device, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt, optimizer, ta_optimizer, scheduler):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_state = copy.deepcopy(get_state_dict(model_s))
    opt.best_acc = 0.0
    opt.best_epoch = 0

    # change teacher model to evaluation mode and set the gradient false
    teacher_i2t.eval()
    teacher_t2i.eval()
    for p in teacher_i2t.parameters():
        p.requires_grad = False
    for p in teacher_t2i.parameters():
        p.requires_grad = False

    # Start training
    for epoch in range(opt.epoch):
        print('epoch {}/{}'.format(epoch + 1, opt.epoch))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            train_loss = OrderedDict({'gen': 0., 'o': 0., 't': 0., 'ta': 0., 'dis': 0., 'c': 0., 'ssl': 0.})
            test_loss = OrderedDict({'gen': 0})

            pbar = tqdm(data_loader[phase], total=len(data_loader[phase]))
            for imgs, origins, txts, labels in pbar:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                # ================forward================
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.

                    if torch.cuda.is_available():
                        origins = origins.float().cuda()
                        texts = txts.clone().float().cuda()
                        imgs = imgs.float().to(output_device)
                        txts = txts.float().to(output_device)
                        labels = labels.to(output_device)

                    if phase == 'eval':
                        # Test the cross-retrieval model in test set.
                        # Calculate the feature and then claculate the Criterion loss.
                        # Please take care that this loss only include the discrimination loss but
                        # without teacher loss.

                        # set model state
                        model_s.eval()

                        # Forward
                        view1_feature, view2_feature, view1_predict, view2_predict, *_ = model_s(origins, texts)

                        loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, opt.phi, opt.alpha, opt.beta)

                        # statistics
                        test_loss['gen'] += loss.item()
                    else:
                        '''
                        Train the model in an adversarial mode
                        '''

                        # ========Discriminator Forward========
                        model_s.eval()
                        tea_ass_img.train()
                        tea_ass_txt.train()

                        for _ in range(opt.dis_iter):
                            # zero the parameter gradients
                            ta_optimizer.zero_grad()

                            # generatarion feature
                            with torch.no_grad():
                                view1_feature, view2_feature, view1_predict, view2_predict, \
                                    stu_img_feats, stu_txt_feats = model_s(origins, texts)

                                tea_img_feats = teacher_i2t.get_embedding(imgs)
                                tea_txt_feats = teacher_t2i.get_embedding(txts)

                                # adversarial loss
                                img_sample_num = stu_img_feats.shape[0]
                                txt_sample_num = stu_txt_feats.shape[0]

                                img_hidden = torch.cat([stu_img_feats, tea_img_feats], dim=0)
                                txt_hidden = torch.cat([stu_txt_feats, tea_txt_feats], dim=0)

                                img_predict = teacher_t2i.get_generation(txt_hidden)
                                txt_predict = teacher_i2t.get_generation(img_hidden)

                            # get training label
                            img_label = torch.cat([torch.zeros(img_sample_num), torch.ones(img_sample_num)]).long()
                            txt_label = torch.cat([torch.zeros(txt_sample_num), torch.ones(txt_sample_num)]).long()
                            img_label = img_label.to(imgs.device)
                            txt_label = txt_label.to(txts.device)

                            # cal loss
                            d_loss = tea_ass_img.train_loss(img_predict, img_label) + tea_ass_txt.train_loss(txt_predict, txt_label)
                            d_loss.backward()
                            ta_optimizer.step()

                            # statistics
                            train_loss['dis'] += d_loss.item() / opt.dis_iter

                        # Generator Forward
                        model_s.train()
                        tea_ass_img.eval()
                        tea_ass_txt.eval()
                        for _ in range(opt.gen_iter):
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            view1_feature, view2_feature, view1_predict, view2_predict, \
                                stu_img_feats, stu_txt_feats = model_s(origins, texts)

                            # get teacher embedding
                            with torch.no_grad():
                                tea_img_feats = teacher_i2t.get_embedding(imgs)
                                tea_txt_feats = teacher_t2i.get_embedding(txts)

                            # old discrimination loss
                            loss_o = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, opt.phi, opt.alpha, opt.beta)

                            # teacher loss
                            loss_t = opt.w_t * (F.mse_loss(stu_img_feats, tea_img_feats) + F.mse_loss(stu_txt_feats, tea_txt_feats))

                            # teaching assistant loss
                            img_predict = teacher_t2i.get_generation(stu_txt_feats)
                            txt_predict = teacher_i2t.get_generation(stu_img_feats)

                            loss_ta = opt.w_ta * (tea_ass_img.adversarial_loss(img_predict) + tea_ass_txt.adversarial_loss(txt_predict))

                            # contrastive loss
                            loss_c = opt.w_c * contrastive_loss(tea_img_feats, tea_txt_feats, stu_img_feats, stu_txt_feats, opt.batch_size, len(data_loader['train']), opt.cll_t) if opt.contrast else torch.tensor(0.)

                            # support set loss
                            loss_ssl = opt.w_ssl * ssl(stu_img_feats, stu_txt_feats, labels, len(data_loader['train']), opt.ssl_t)

                            # backward + optimize only if in training phase
                            loss = loss_o + loss_t + loss_ta + loss_c + loss_ssl

                            loss.backward()
                            optimizer.step()

                            # statistics
                            train_loss['gen'] += loss.item() / opt.gen_iter
                            train_loss['o'] += loss_o.item() / opt.gen_iter
                            train_loss['t'] += loss_t.item() / opt.gen_iter
                            train_loss['ta'] += loss_ta.item() / opt.gen_iter
                            train_loss['c'] += loss_c.item() / opt.gen_iter
                            train_loss['ssl'] += loss_ssl.item() / opt.gen_iter

            # report train or evaluate retults
            report = train_loss if phase == 'train' else test_loss
            epoch_loss = report['gen'] / len(data_loader[phase])
            for key, value in report.items():
                print('  {}:{:.2f}'.format(key, value / len(data_loader[phase])), end=',')
            print()

            # Evaluation in the according dataset
            eval_i2t, eval_t2i = eval_vgg_model_map(model_s, data_loader[phase], sample_num=1000)
            print('{} loss: {:.4f}, {} dataset eval result - img2txt: {:.4f}  txt2img: {:.4f}'.format(phase, epoch_loss, phase, eval_i2t, eval_t2i))

            # record evaluate results
            if phase == 'eval':
                test_img_acc_history.append(eval_i2t)
                test_txt_acc_history.append(eval_t2i)
                epoch_loss_history.append(epoch_loss)
                # save best record
                if (eval_i2t + eval_t2i) / 2. > opt.best_acc:
                    opt.best_acc = (eval_i2t + eval_t2i) / 2.
                    opt.best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(get_state_dict(model_s))
                # lr scheduler update after evaluate
                if scheduler is not None:
                    scheduler.step((eval_i2t + eval_t2i) / 2.)
        print('>>> current best accuracy: {:.6f}, best epoch: {}\n'.format(opt.best_acc, opt.best_epoch))

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best average accuracy: {:4f}'.format(opt.best_acc))
    print('best epoch: {}'.format(opt.best_epoch))

    # load best model weights
    if isinstance(model_s, DataParallel):
        model_s.module.load_state_dict(best_model_state)
    else:
        model_s.load_state_dict(best_model_state)

    return test_img_acc_history, test_txt_acc_history, epoch_loss_history


def eval_vgg_model_map(model, dataloader, sample_num=None):
    # eval retrieval score in train and val set
    model.eval()

    t_imgs, t_txts, t_labels = [], [], []

    with torch.no_grad():
        for imgs, origins, txts, labels in dataloader:
            if torch.cuda.is_available():
                origins = origins.float().cuda()
                txts = txts.float().cuda()

            t_view1_feature, t_view2_feature, *_ = model(origins, txts)
            t_imgs.append(t_view1_feature.cpu().numpy())
            t_txts.append(t_view2_feature.cpu().numpy())
            t_labels.append(labels)

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)

    eval_i2t = fx_calc_map_label(t_imgs, t_txts, t_labels, sample_num=sample_num)
    eval_t2i = fx_calc_map_label(t_txts, t_imgs, t_labels, sample_num=sample_num)

    return eval_i2t, eval_t2i


def get_state_dict(model):
    """
    Utiles function to get state_dict in a right way
    """

    if isinstance(model, DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()