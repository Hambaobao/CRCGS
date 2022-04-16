from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models

from .model import ImgNN, TextNN


def convert_dim(input_dim: int, hidden_dim: list, output_dim: int) -> list:
    '''
    Convert the input, hidden, output dims to a unified format
    '''

    dim_list = [input_dim] + hidden_dim + [output_dim]
    return dim_list


def mlp_network(dim_list, activation='Tanh') -> nn.Module:
    '''
    Build a Multi-layer Perceptron Neural Network with specific activation function
    '''
    assert len(dim_list) > 1

    first_dim_list, second_dim_list = dim_list[:-1], dim_list[1:]
    args_list = zip(first_dim_list, second_dim_list)
    activation = getattr(nn, activation)
    model_list = [[nn.Linear(in_dim, out_dim), activation()]
                  for in_dim, out_dim in args_list]
    model = nn.Sequential(*list(reduce(lambda x, y: x + y, model_list)))
    return model


class Discriminator(nn.Module):
    """
    Used for Adversarial Training
    """

    def __init__(self, input_dim=4096, hidden_dim=[1024, 512], output_dim=64):
        super(Discriminator, self).__init__()
        dim_list = convert_dim(input_dim, hidden_dim, output_dim)
        self.mlp = mlp_network(dim_list, activation='ReLU')
        self.classifier = nn.Linear(output_dim, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        mid = self.mlp(x)
        logit = self.classifier(mid)
        return logit

    def train_loss(self, x, label):
        logit = self(x)
        loss = self.criterion(logit, label)
        return loss

    def adversarial_loss(self, x):
        logit = self(x)
        label = torch.ones(x.size(0)).long().to(x.device)
        loss = self.criterion(logit, label)
        return loss


class ConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        sub = input - target
        sub = sub ** 2
        loss = torch.sum(sub, dim=-1)
        return getattr(torch, self.reduction)(loss)


class VGGNet(nn.Module):
    '''
    VGG11: A shallow version of VGGNet
    '''

    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg11_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(
            *list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TextCNN(nn.Module):
    '''
    Simple implementation TextCNN
    '''

    def __init__(self, embedding, kernel_num, kernel_sizes, dropout=0.1, num_categories=3):
        super(TextCNN, self).__init__()
        self.embedding = embedding
        embed_size = embedding.embedding_dim
        self.conv_layers = nn.ModuleList()
        representation_size = 0
        for kernel_size in kernel_sizes:
            self.conv_layers.append(
                nn.Conv2d(1, kernel_num, (kernel_size, embed_size)))
            representation_size += kernel_num
        self.dropout = dropout
        self.output_projection = nn.Linear(representation_size, num_categories)

    def forward(self, sentence, aspect):
        sentence = self.embedding(sentence).unsqueeze(1)
        sentence = F.dropout(sentence, p=self.dropout, training=self.training)
        representation = []

        for conv in self.conv_layers:
            y = F.relu(conv(sentence).squeeze(3))
            y = F.max_pool1d(y, y.size(2)).squeeze(2)
            representation.append(y)
        representation = torch.cat(representation, dim=1)
        logit = self.output_projection(representation)
        return logit


class IDCM_NN(nn.Module):
    """
    Network to learn text representations
    """

    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.text_net = TextNN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict


class EncoderToDecoder(nn.Module):
    """
    A Base Object for EncoderToDecoder
    """

    def __init__(self, encoder, decoder, criterion, training=True, path=None):
        super(EncoderToDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.common_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.data_assignment = lambda x, y, z: (x, y, z)

        # prepare for training
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if training:
            self.training_init = training
            self.reset()
        if path:
            print('Start from: ', path)
            self.load_checkpoint(path)
        self.to(self.device)

    def forward(self, x):
        mid = self.encoder(x)
        y = self.decoder(mid)
        return y

    def get_embedding(self, x):
        mid = self.encoder(x)
        return mid

    def get_generation(self, mid):
        y = self.decoder(mid)
        return y

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path + 'model.pth')
        torch.save(self.optimizer.state_dict(), path + 'optimizer.pth')
        torch.save(self.scheduler.state_dict(), path + 'scheduler.pth')

    def load_checkpoint(self, path):
        state_dict = torch.load(path + 'model.pth', map_location=self.device)

        # filter the discriminator component in the load state dict
        filtered = {}
        for key in state_dict:
            if 'discriminator' in key:
                continue
            filtered[key] = state_dict[key]
        self.load_state_dict(filtered)

        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(torch.load(
                path + 'optimizer.pth', map_location=self.device))
        if hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(torch.load(
                path + 'scheduler.pth', map_location=self.device))

        self.to(self.device)

    def train_loss(self, x, y):
        pred_y = self(x)
        loss = self.criterion(pred_y, y)
        return loss

    def train_prepare(self, optimizer, lr, scheduler, patience, mode, grad_clip, data_assignment=None):
        optimizer = getattr(optim, optimizer.capitalize())
        scheduler = getattr(lr_scheduler, scheduler)
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.scheduler = scheduler(
            self.optimizer, mode=mode, factor=0.5, patience=patience, min_lr=0.000001, verbose=True)
        self.grad_clip = lambda: nn.utils.clip_grad_norm_(
            self.parameters(), grad_clip)

        if data_assignment:
            self.data_assignment = data_assignment

    def reset(self):
        self.loss, self.print_every = 0, 1

    def print_loss(self):
        print_loss = self.loss / self.print_every
        self.print_every += 1
        return 'Loss:{:.2f}'.format(print_loss)


class TextEncoder(nn.Module):
    """
    Used for Text2Image Training
    """

    def __init__(self, input_dim=1024, hidden_dim: list = [1024, 512], output_dim=1024, dropout_rate=0.5):
        super(TextEncoder, self).__init__()
        dim_list = convert_dim(input_dim, hidden_dim, output_dim)
        self.encoder = mlp_network(dim_list, activation='ReLU')
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        return self.encoder(x)


class ImgEncoder(nn.Module):
    """
    Used for Image2Text Training
    """

    def __init__(self, input_dim=4096, hidden_dim: list = [1024, 512], output_dim=1024, dropout_rate=0.5):
        super(ImgEncoder, self).__init__()
        dim_list = convert_dim(input_dim, hidden_dim, output_dim)
        self.encoder = mlp_network(dim_list, activation='ReLU')
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        return self.encoder(x)


class TextDecoder(nn.Module):
    """
    Used for Image2Text Training
    """

    def __init__(self, input_dim=1024, hidden_dim: list = [1024, 512],  output_dim=4096):
        super(TextDecoder, self).__init__()
        dim_list = convert_dim(input_dim, hidden_dim, output_dim)
        self.decoder = mlp_network(dim_list, activation='ReLU')

    def forward(self, x):
        return self.decoder(x)


class ImgDecoder(nn.Module):
    """
    Used for Text2Image Training
    """

    def __init__(self, input_dim=1024, hidden_dim: list = [1024, 512], output_dim=4096):
        super(ImgDecoder, self).__init__()
        dim_list = convert_dim(input_dim, hidden_dim, output_dim)
        self.decoder = mlp_network(dim_list, activation='ReLU')

    def forward(self, x):
        return self.decoder(x)


class Img2Text(EncoderToDecoder):
    """
    A Wrapper for Image2Text Generation Task
    """

    def __init__(self, input_dim: int, input2mid_dim: list, mid_dim: int, mid2output_dim: list, output_dim: int, dropout_rate=0.5, path=None):
        img_encoder = ImgEncoder(
            input_dim=input_dim,
            hidden_dim=input2mid_dim,
            output_dim=mid_dim,
            dropout_rate=dropout_rate
        )
        text_decoder = TextDecoder(
            input_dim=mid_dim,
            hidden_dim=mid2output_dim,
            output_dim=output_dim
        )
        criterion = nn.L1Loss()
        self.hidden_dim = mid_dim
        super(Img2Text, self).__init__(
            img_encoder, text_decoder, criterion, path=path)


class Text2Img(EncoderToDecoder):
    """
    A Wrapper for Text2Img Generation Task
    """

    def __init__(self, input_dim: int, input2mid_dim: list, mid_dim: int, mid2output_dim: list, output_dim: int, dropout_rate=0.5, path=None):
        text_encoder = TextEncoder(
            input_dim=input_dim,
            hidden_dim=input2mid_dim,
            output_dim=mid_dim,
            dropout_rate=dropout_rate
        )
        img_decoder = ImgDecoder(
            input_dim=mid_dim,
            hidden_dim=mid2output_dim,
            output_dim=output_dim
        )
        criterion = nn.L1Loss()
        self.hidden_dim = mid_dim
        super(Text2Img, self).__init__(
            text_encoder, img_decoder, criterion, path=path)


class BinaryLayer(nn.Module):
    """
    Hash continuous feature vector into the binary format.
    """
    def __init__(self, d_emb, d_code):
        super(BinaryLayer, self).__init__()
        self.embedding_dropout = nn.Dropout(0.3)

        # hash
        self.linear1 = nn.Linear(d_emb, d_emb // 2)
        self.linear2 = nn.Linear(d_emb // 2, d_emb // 2)
        self.hash_layer = nn.Conv1d(1, d_code, kernel_size=d_emb // 2)

    def forward(self, x):
        x = self.embedding_dropout(x)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = x.unsqueeze(1)
        x = self.hash_layer(x)
        x = x.squeeze(-1)
        x = torch.tanh(x)
        return x
