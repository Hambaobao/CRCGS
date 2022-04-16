import torch
import torch.optim as optim
from torchvision import transforms

from src.load_data import get_coco_loader, get_flickr_vgg_loader, get_nus_vgg_loader
from src.model import VGGIDCM_NN, IDCM_NN
from src.my_model import BinaryLayer
from src.train_binary_model import eval_model_binary_map, train_model_binary
from src.train_binary_model import eval_model_binary_map_coco, train_model_binary_coco

from utils.parse import parse_binary_option

import os


def build_image_transform():
    # build image transform
    totensor = transforms.Lambda(lambda x: torch.from_numpy(x))
    scale = transforms.Lambda(lambda x: x / 255.)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([totensor, scale, normalize])

    return transform


def load_data(opt, transform):
    # dataloader setting
    print('>>> Data loading is beginning <<<')
    if opt.dataset == 'coco':
        data_loader, data_parameter = get_coco_loader(opt.data_dir, opt.batch_size)
    elif opt.dataset == 'flickr':
        data_loader, data_parameter = get_flickr_vgg_loader(opt.data_dir, opt.batch_size, pre_transform=transform)
    elif opt.dataset == 'nus':
        data_loader, data_parameter = get_nus_vgg_loader(opt.data_dir, opt.batch_size, pre_transform=transform)
    else:
        raise Exception('Dataset did not exist...')
    print('=== Data loading is completed ===\n')
    print('>>> Running model in {} dataset <<<'.format(opt.dataset))

    return data_loader, data_parameter


def build_models(opt, data_parameter):
    # build model
    # student
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        extract_model = VGGIDCM_NN(img_input_dim=data_parameter['img_dim'], img_output_dim=2048, txt_input_dim=data_parameter['txt_dim'], txt_output_dim=2048, output_dim=data_parameter['num_class'], minus_one_dim=opt.sem_emb_size)
    elif opt.dataset == 'coco':
        extract_model = IDCM_NN(img_input_dim=data_parameter['img_dim'], img_output_dim=2048, txt_input_dim=data_parameter['txt_dim'], txt_output_dim=2048, output_dim=data_parameter['num_class'], minus_one_dim=opt.sem_emb_size)

    binary_model = BinaryLayer(1024, d_code=opt.code_len)

    extract_model.load_state_dict(torch.load(opt.extract_model_file))

    return extract_model, binary_model


def record_results(binary_model, opt, eval_i2t, eval_t2i):
    # save best model weights
    best_model_state = binary_model.state_dict()

    accuracy = (eval_i2t + eval_t2i) / 2.
    save_file = os.path.join(opt.save_folder, 'binary{}_model_{:.4f}.pt'.format(opt.code_len, accuracy))
    torch.save(best_model_state, save_file)
    print(">>> model file save to: {}".format(save_file))

    # record and show results
    save_file = os.path.join(opt.save_folder, 'binary{}_record_{:.4f}.txt'.format(opt.code_len, accuracy))
    with open(save_file, 'w') as f:
        # results
        s1 = '>>>> Image to Text MAP = {}'.format(eval_i2t)
        s2 = '>>>> Text to Image MAP = {}'.format(eval_t2i)
        s3 = '>>>> Average MAP = {}'.format(accuracy)
        f.write('\n'.join([s1, s2, s3]) + '\n')

        # setting
        f.write('\n===== parameters =====\n\n')
        for key, value in opt.__dict__.items():
            f.write('{}:{}\n'.format(key, value))

    # show results
    print(s1, s2, s3, sep='\n')
    print(">>> results file save to: {}".format(save_file))

    return accuracy


def main(opt):
    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.CUDA)[1:-1]

    # GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = build_image_transform()

    # dataloader setting
    data_loader, data_parameter = load_data(opt, transform)

    # build models
    extract_model, binary_model = build_models(opt, data_parameter)

    # build optimizer
    params_to_update = list(binary_model.parameters())
    optimizer = optim.Adam(params_to_update, lr=opt.lr, betas=opt.betas)

    # scheduler
    scheduler = None

    # to GPU
    extract_model.to(device)
    binary_model.to(device)

    # train and evaluate model
    print('>>> Training is beginning <<<')
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        train_model_binary(opt, binary_model, extract_model, data_loader, optimizer, 'cuda', scheduler)
    elif opt.dataset == 'coco':
        train_model_binary_coco(opt, binary_model, extract_model, data_loader, optimizer, 'cuda', scheduler)
    print('>>> Training is completed <<<\n')

    # evaluate
    print('>>> Evaluatiing on test data <<<')
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        eval_i2t, eval_t2i = eval_model_binary_map(extract_model, binary_model, data_loader['test'])
    elif opt.dataset == 'coco':
        eval_i2t, eval_t2i = eval_model_binary_map_coco(extract_model, binary_model, data_loader['test'])

    # record and show results
    accuracy = record_results(binary_model, opt, eval_i2t, eval_t2i)

    return accuracy


if __name__ == "__main__":
    # parameters
    opt = parse_binary_option()

    main(opt)