import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys

sys.path.append('..')
from src.model import VGGIDCM_NN
from src.my_model import BinaryLayer
from utils.cal_result import cal_map_results
from utils.parse import parse_evaluate_option

import os


class CustomDataSetWithTrans(Dataset):

    def __init__(self, images, texts, labels, sample_index, transform=None):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.sample_ind = sample_index
        self.transform = transform

    def __getitem__(self, index):
        index = self.sample_ind[index]

        text = self.texts[index]
        label = self.labels[index]

        img = self.images[index]
        if self.transform is not None:
            img = img.transpose((0, 2, 1))
            img = self.transform(img)
        return img, text, label

    def __len__(self):
        count = len(self.sample_ind)
        assert len(self.images) == len(self.labels)
        return count


def get_data(legal_data_list, test_data_list, retrieval_data_list=None):
    """
    Get image and text from query list and retreival list
    """

    query_list = test_data_list
    if retrieval_data_list is None:
        retrieval_list = np.setdiff1d(legal_data_list, query_list)
    else:
        retrieval_list = retrieval_data_list

    return query_list, retrieval_list


def evaluate(opt):
    # load data
    images = h5py.File('/data0/zl/cross-modal/process_data/nus/nus-wide-tc21-iall.mat', mode='r')
    images = images['IAll']
    texts = np.load('/data0/zl/cross-modal/process_data/nus/bow.npy')
    labels = np.load('/data0/zl/cross-modal/process_data/nus/label.npy')
    test_list = np.load('/data0/zl/cross-modal/process_data/nus/test_data_list.npy')
    legal_list = np.load('/data0/zl/cross-modal/process_data/nus/legal_data_list.npy')

    # model file
    model_path = '/data0/zl/cross-modal/student_model_save/nus/model.pt'
    binary_path = '/data0/zl/cross-modal/binary_model_save/nus/binary{}_model.pt'.format(opt.code_len)

    # set parameters
    semantic_embedding_size = 1024
    code_length = opt.code_len
    binary = True

    # get query and retrieval set
    query_ind, retrieval_ind = get_data(legal_list, test_list)

    # build dataloader
    totensor = transforms.Lambda(lambda x: torch.from_numpy(x))
    scale = transforms.Lambda(lambda x: x / 255.)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transfrom = transforms.Compose([totensor, scale, normalize])

    # query
    query_dataset = CustomDataSetWithTrans(images, texts, labels, query_ind, transfrom)
    query_loader = DataLoader(query_dataset, batch_size=100, shuffle=False)

    # retrieve
    retrieval_dataset = CustomDataSetWithTrans(images, texts, labels, retrieval_ind, transfrom)
    retrieval_loader = DataLoader(retrieval_dataset, batch_size=100, shuffle=False)

    # build student model
    stu_model = VGGIDCM_NN(img_input_dim=4096, img_output_dim=2048, txt_input_dim=texts.shape[-1], txt_output_dim=2048, output_dim=labels.shape[-1], minus_one_dim=semantic_embedding_size).cuda()

    # build binary model
    bin_model = BinaryLayer(semantic_embedding_size, code_length).cuda()

    # load student model
    print('>>> load student model <<<')
    stu_model.load_state_dict(torch.load(model_path))

    # load binary model
    print('>>> load binary model <<<')
    bin_model.load_state_dict(torch.load(binary_path))

    # test
    eval_i2t, eval_t2i = cal_map_results(stu_model, query_loader, retrieval_loader, bin_model, binary)
    accuracy = (eval_i2t + eval_t2i) / 2.

    # record results
    save_folder = '/data0/zl/cross-modal/final_results/nus/'
    save_file = os.path.join(save_folder, 'binary{}_record_{:.4f}.txt'.format(opt.code_len, accuracy))
    with open(save_file, 'w') as f:
        # results
        s1 = '>>> Image to Text MAP = {}'.format(eval_i2t)
        s2 = '>>> Text to Image MAP = {}'.format(eval_t2i)
        s3 = '>>> Average MAP = {}'.format(accuracy)
        f.write('\n'.join([s1, s2, s3]) + '\n')

    # show results
    print(s1, s2, s3, sep='\n')
    print(">>> results file save to: {}".format(save_file))


if __name__ == '__main__':
    # parameters
    opt = parse_evaluate_option()

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.CUDA)[1:-1]

    evaluate(opt)