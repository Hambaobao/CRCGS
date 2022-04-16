import torch
import torch.nn as nn

eps = 1e-7


def normalize(matrix, tea_feats, stu_feats):
    # shape [batch_size, 1]
    tea_norm = torch.norm(tea_feats, dim=1).view(matrix.size(0), -1)

    # shape [1, batch_size]
    stu_norm = torch.norm(stu_feats, dim=1).view(-1, matrix.size(1))

    # shape [batch_size]
    norm = tea_norm * stu_norm

    # normalize
    matrix_n = torch.div(matrix, norm + eps)

    return matrix_n


def calc_loss(tea_feats, stu_feats, batch_size, ndata, T=0.07):
    # similarity matrix shape [batch_size, batch_size]
    # similar_matrix = torch.div(torch.mm(tea_feats, stu_feats.t()), batch_size)
    similar_matrix = torch.mm(tea_feats, stu_feats.t())

    # normalized similarity matrix
    similar_matrix_n = normalize(similar_matrix, tea_feats, stu_feats)

    similar_matrix_t = torch.div(similar_matrix_n, T)

    similar_matrix_e = torch.exp(similar_matrix_t)

    # number of negatives paired with each positive
    m = batch_size - 1

    # noise distribution
    Pn = 1 / float(ndata)

    # postive pairs shape [batch_size]
    pos_pairs = torch.diag(similar_matrix_e)

    log_D1 = torch.div(pos_pairs, pos_pairs.add(m * Pn + eps)).log_()

    # loss for K negative pair
    log_D0_p = torch.div(pos_pairs.clone().fill_(m * Pn), pos_pairs.add(m * Pn + eps)).log_()

    log_D0 = torch.div(similar_matrix_e.clone().fill_(m * Pn), similar_matrix_e.add(m * Pn + eps)).log_()

    loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0) - log_D0_p.view(-1, 1).sum(0)) / batch_size

    return loss


def contrastive_loss(tea_img_feats, tea_txt_feats, stu_img_feats, stu_txt_feats, batch_size, ndata, T=0.07):
    # img loss
    img_loss = calc_loss(tea_img_feats, stu_img_feats, batch_size, ndata, T)

    # txt loss
    txt_loss = calc_loss(tea_txt_feats, stu_txt_feats, batch_size, ndata, T)

    # cross loss
    cross_loss_0 = calc_loss(tea_img_feats, stu_txt_feats, batch_size, ndata, T)

    cross_loss_1 = calc_loss(tea_txt_feats, stu_img_feats, batch_size, ndata, T)

    return (img_loss + txt_loss + cross_loss_0 + cross_loss_1) / 4.


def build_categorys(labels):
    # build category with labels
    # labels.shape  [batch_size, 21]
    categorys = []

    for index in range(labels.shape[1]):
        category = []
        for n, l in enumerate(labels):
            if l[index]:
                category.append(n)
        categorys.append(category)

    return categorys


def get_sup_feat(n, label, img_feats, txt_feats, categorys, T):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    feat_length = img_feats.size()[1]

    img_sup_feat = torch.zeros(feat_length).to(img_feats.device)
    txt_sup_feat = torch.zeros(feat_length).to(txt_feats.device)

    img_denomi = 1e-6
    txt_denomi = 1e-6

    for i, tag in enumerate(label):
        if tag:
            for k in categorys[i]:
                img_denomi = img_denomi + torch.exp(cos(img_feats[n], txt_feats[k]) / T)
                txt_denomi = txt_denomi + torch.exp(cos(txt_feats[n], img_feats[k]) / T)

            for j in categorys[i]:
                img_sup_feat = img_sup_feat + (torch.exp(cos(img_feats[n], txt_feats[j]) / T)) / img_denomi * txt_feats[j]
                txt_sup_feat = txt_sup_feat + (torch.exp(cos(txt_feats[n], img_feats[j]) / T)) / txt_denomi * img_feats[j]

    return img_sup_feat, txt_sup_feat


def get_support_feats(stu_img_feats, stu_txt_feats, labels, T):
    labels = labels.float()

    # similarity matrix shape [batch_size, batch_size]
    similar_matrix = torch.mm(stu_img_feats, stu_txt_feats.t())

    # normalized similarity matrix
    similar_matrix_n = normalize(similar_matrix, stu_img_feats, stu_txt_feats)

    similar_matrix_t = torch.div(similar_matrix_n, T)

    similar_matrix_e = torch.exp(similar_matrix_t)

    # get denominators of weight
    img_denomis = torch.diag(torch.mm(torch.mm(similar_matrix_e, labels), labels.t())).reshape(-1, 1)

    txt_denomis = torch.diag(torch.mm(torch.mm(similar_matrix_e.t(), labels), labels.t())).reshape(-1, 1)

    label_matrix = torch.zeros(labels.shape[0], labels.shape[0]).to(labels.device)

    # build label matrix shape [batch_size, batch_size], which is symmetric
    for i, l in enumerate(labels):
        for j, tag in enumerate(l):
            if tag:
                for k, _ in enumerate(labels):
                    if labels[k][j]:
                        label_matrix[i][k] += 1

    img_label_matrix = torch.mul(similar_matrix_e, label_matrix)
    txt_label_matrix = torch.mul(similar_matrix_e.t(), label_matrix)

    img_sup_feats = torch.div(torch.mm(img_label_matrix, stu_txt_feats), img_denomis)
    txt_sup_feats = torch.div(torch.mm(txt_label_matrix, stu_img_feats), txt_denomis)

    return img_sup_feats, txt_sup_feats


def cacl_support_set_loss(stu_img_feats, stu_txt_feats, img_sup_feats, txt_sup_feats):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    img_sim = cos(stu_img_feats, img_sup_feats)
    txt_sim = cos(stu_txt_feats, txt_sup_feats)

    ss_loss = -torch.log((torch.sum(img_sim) / stu_img_feats.shape[0] + torch.sum(txt_sim) / stu_txt_feats.shape[0]) / 2.)

    return ss_loss


def support_set_loss(stu_img_feats, stu_txt_feats, labels, ndata, T):
    img_sup_feats, txt_sup_feats = get_support_feats(stu_img_feats, stu_txt_feats, labels, T)

    ss_loss = cacl_support_set_loss(stu_img_feats, stu_txt_feats, img_sup_feats, txt_sup_feats)

    return ss_loss