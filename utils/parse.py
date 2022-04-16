import argparse


def parse_option():
    """tranning parameters setting"""

    parser = argparse.ArgumentParser('argument for training')

    # CUDA
    parser.add_argument('--CUDA', type=int, default=0, nargs='+', help='CUDA_VISIBLE_DEVICES')

    # dadaset
    parser.add_argument('--dataset', type=str, default='nus', choices=['nus', 'coco', 'flickr'], help='dataset')

    # train parameters
    parser.add_argument('--alpha', type=float, default=1, help='alpha: used in calculate loss')
    parser.add_argument('--beta', type=float, default=0.001, help='beta: used in calculate loss')
    parser.add_argument('--phi', type=float, default=1, help='phi: used in calculate loss')

    parser.add_argument('--w_t', type=float, default=0.2, help='weight of teacher loss')

    parser.add_argument('--w_ta', type=float, default=0.01, help='weight of teacher assitant loss')

    parser.add_argument('--w_c', type=float, default=0.03, help='weight of embedding level contrastive loss')
    parser.add_argument('--w_ssl', type=float, default=40., help='weight of support set loss')

    parser.add_argument('--gen_iter', type=int, default=1, help='number of generation iteration')
    parser.add_argument('--dis_iter', type=int, default=3, help='number of discriminate iteration')
    parser.add_argument('--epoch', type=int, default=80, help='epoch')

    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--sem_emb_size', type=int, default=1024, help='semantic embedding size')

    parser.add_argument('--parallel', type=bool, default=True, help='parallelly train model')
    parser.add_argument('--contrast', type=bool, default=False, help='enable contrastive loss')

    # optimization
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--betas', type=float, default=(0.5, 0.999), nargs='+', help='used in optimizer')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')

    parser.add_argument('--cll_t', default=0.07, type=float, help='temperature parameter for contrastive learning loss')

    parser.add_argument('--ssl_t', default=0.07, type=float, help='temperature parameter for support set loss')

    opt = parser.parse_args()

    # file path
    opt.teacher_i2t_path = '/data0/zl/cross-modal/teacher_checkpoints/{}/Img2Text/'.format(opt.dataset)
    opt.teacher_t2i_path = '/data0/zl/cross-modal/teacher_checkpoints/{}/Text2Img/'.format(opt.dataset)
    opt.data_dir = '/data0/zl/cross-modal/process_data/{}'.format(opt.dataset)
    opt.save_folder = '/data0/zl/cross-modal/student_model_save/{}/'.format(opt.dataset)

    return opt


def parse_binary_option():
    """tranning parameters setting"""

    parser = argparse.ArgumentParser('argument for binary layer training')

    # CUDA
    parser.add_argument('--CUDA', type=int, default=[0, 1], nargs='+', help='CUDA_VISIBLE_DEVICES')

    # dadaset
    parser.add_argument('--dataset', type=str, default='nus', choices=['nus', 'coco', 'flickr'], help='dataset')

    # train parameters
    parser.add_argument('--alpha', type=float, default=1, help='alpha: used in calculate loss')
    parser.add_argument('--beta', type=float, default=1, help='beta: used in calculate loss')
    parser.add_argument('--gamma', type=float, default=0, help='phi: used in calculate loss')

    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--code_len', type=int, default=32, help='binary code length')

    parser.add_argument('--sem_emb_size', type=int, default=1024, help='semantic embedding size')

    # optimization
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--betas', type=float, default=(0.5, 0.999), nargs='+', help='used in optimizer')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')

    opt = parser.parse_args()

    # file path
    opt.data_dir = '/data0/zl/cross-modal/process_data/{}'.format(opt.dataset)
    opt.extract_model_file = '/data0/zl/cross-modal/student_model_save/{}/best_results/model.pt'.format(opt.dataset)
    opt.save_folder = '/data0/zl/cross-modal/binary_model_save/{}/'.format(opt.dataset)

    return opt


def parse_evaluate_option():
    parser = argparse.ArgumentParser('argument for evaluating')

    # CUDA
    parser.add_argument('--CUDA', type=int, default=[0, 1], nargs='+', help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--code_len', type=int, default=32, help='binary code length')

    opt = parser.parse_args()

    return opt