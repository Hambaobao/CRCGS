import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from torchvision import transforms
import torch.backends.cudnn as cudnn

from utils.parse import parse_option

from src.model import VGGIDCM_NN, IDCM_NN
from src.my_model import Discriminator, Img2Text, Text2Img
from src.load_data import get_coco_loader, get_flickr_vgg_loader, get_nus_vgg_loader
from src.train_model import train_model_with_vgg, eval_vgg_model_map, train_model, eval_model_map

import os


def GPU_setting(opt):
    # set GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.parallel:
        output_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        output_device = device

    return device, output_device


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
        model_s = VGGIDCM_NN(img_input_dim=data_parameter['img_dim'], img_output_dim=2048, txt_input_dim=data_parameter['txt_dim'], txt_output_dim=2048, output_dim=data_parameter['num_class'], minus_one_dim=opt.sem_emb_size)
    elif opt.dataset == 'coco':
        model_s = IDCM_NN(img_input_dim=data_parameter['img_dim'], img_output_dim=2048, txt_input_dim=data_parameter['txt_dim'], txt_output_dim=2048, output_dim=data_parameter['num_class'], minus_one_dim=opt.sem_emb_size)

    # teacher
    if opt.dataset == 'nus':
        teacher_i2t = Img2Text(4096, [2048], 1024, [2000], 1000, 0.3, opt.teacher_i2t_path)
        teacher_t2i = Text2Img(1000, [2048], 1024, [2048], 4096, 0.3, opt.teacher_t2i_path)
    elif opt.dataset == 'flickr':
        teacher_i2t = Img2Text(4096, [2048], 1024, [2000], 1386, 0.3, opt.teacher_i2t_path)
        teacher_t2i = Text2Img(1386, [2048], 1024, [2048], 4096, 0.3, opt.teacher_t2i_path)
    elif opt.dataset == 'coco':
        teacher_i2t = Img2Text(4096, [2048], 1024, [2000], 2000, 0.3, opt.teacher_i2t_path)
        teacher_t2i = Text2Img(2000, [2048], 1024, [2048], 4096, 0.3, opt.teacher_t2i_path)

    # teacher assistant
    if opt.dataset == 'nus':
        tea_ass_img = Discriminator(4096, [2048, 1024], 512)
        tea_ass_txt = Discriminator(1000, [512, 64], 32)
    elif opt.dataset == 'flickr':
        tea_ass_img = Discriminator(4096, [2048, 1024], 512)
        tea_ass_txt = Discriminator(1386, [512, 64], 32)
    elif opt.dataset == 'coco':
        tea_ass_img = Discriminator(4096, [2048, 1024], 512)
        tea_ass_txt = Discriminator(2000, [512, 64], 32)

    return model_s, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt


def record_results(model_s, opt, eval_i2t, eval_t2i):
    # save best model weights
    best_model_state = model_s.module.state_dict() if isinstance(model_s, DataParallel) else model_s.state_dict()

    accuracy = (eval_i2t + eval_t2i) / 2.
    save_file = os.path.join(opt.save_folder, 'model_{:.4f}.pt'.format(accuracy))
    torch.save(best_model_state, save_file)
    print(">>> model file save to: {}".format(save_file))

    # record and show results
    save_file = os.path.join(opt.save_folder, 'record_{:.4f}.txt'.format(accuracy))
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
    device, output_device = GPU_setting(opt)

    # build image transform
    transform = build_image_transform()

    # dataloader setting
    data_loader, data_parameter = load_data(opt, transform)

    # build models
    model_s, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt = build_models(opt, data_parameter)

    # build optimizer
    params_to_update = list(model_s.parameters())
    optimizer = optim.Adam(params_to_update, lr=opt.lr, betas=opt.betas)

    ta_params_to_update = list(tea_ass_img.parameters()) + list(tea_ass_txt.parameters())
    ta_optimizer = optim.Adam(ta_params_to_update, lr=opt.lr, betas=tuple(opt.betas))

    # build scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True) if opt.dataset == 'coco' else None

    # to GPU
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        if torch.cuda.is_available():
            model_s.cuda()
            teacher_t2i.to(output_device)
            teacher_i2t.to(output_device)
            tea_ass_img.to(output_device)
            tea_ass_txt.to(output_device)
            cudnn.benchmark = True
            if opt.parallel:
                model_s = DataParallel(model_s, device_ids=[device, output_device], output_device=output_device)
    elif opt.dataset == 'coco':
        if torch.cuda.is_available():
            model_s.cuda()
            teacher_t2i.to(device)
            teacher_i2t.to(device)
            tea_ass_img.to(device)
            tea_ass_txt.to(device)
            cudnn.benchmark = True

    # train student
    print('>>> Training is beginning <<<')
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        train_model_with_vgg(opt, data_loader, model_s, output_device, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt, optimizer, ta_optimizer, scheduler)
    elif opt.dataset == 'coco':
        train_model(opt, data_loader, model_s, output_device, teacher_i2t, teacher_t2i, tea_ass_img, tea_ass_txt, optimizer, ta_optimizer, scheduler)
    print('=== Training is completed ===\n')

    # evaluate
    print('>>> Evaluation on test data <<<')
    if opt.dataset == 'nus' or opt.dataset == 'flickr':
        eval_i2t, eval_t2i = eval_vgg_model_map(model_s, data_loader['test'])
    elif opt.dataset == 'coco':
        eval_i2t, eval_t2i = eval_model_map(model_s, data_loader['test'])

    # record and show results
    accuracy = record_results(model_s, opt, eval_i2t, eval_t2i)

    return accuracy


if __name__ == '__main__':
    # parameters
    opt = parse_option()

    main(opt)