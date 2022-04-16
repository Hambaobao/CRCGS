import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(
            *list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        self.denseL2 = nn.Linear(output_dim, output_dim)
        self.denseL3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.denseL1(x))
        x = torch.tanh(self.denseL2(x))
        out = torch.tanh(self.denseL3(x))
        return out


class VGGImgNN(nn.Module):
    """Img Net base on fintuning VGG Net"""

    def __init__(self, input_dim, output_dim):
        super(VGGImgNN, self).__init__()
        self.extractor = VGGNet()
        self.img = ImgNN(input_dim, output_dim)

    def forward(self, x):
        x = self.extractor(x)
        out = self.img(x)
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        self.denseL2 = nn.Linear(output_dim, output_dim)
        self.denseL3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.denseL1(x))
        x = torch.tanh(self.denseL2(x))
        out = torch.tanh(self.denseL3(x))
        return out


class IDCM_NN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, img_input_dim=4096, img_output_dim=1024,
                 txt_input_dim=1024, txt_output_dim=1024, minus_one_dim=1024, output_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.txt_net = TextNN(txt_input_dim, txt_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

        self.img_layer = nn.Linear(minus_one_dim, 1024)
        self.txt_layer = nn.Linear(minus_one_dim, 1024)

    def forward(self, img, txt):
        view1_feature = self.img_net(img)
        view2_feature = self.txt_net(txt)
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)

        stu_img_feature = F.relu(self.img_layer(view1_feature))
        stu_txt_feature = F.relu(self.txt_layer(view2_feature))

        return view1_feature, view2_feature, view1_predict, view2_predict, stu_img_feature, stu_txt_feature


class VGGIDCM_NN(nn.Module):
    """Network to learn text representations base on fintuning VGG Net"""

    def __init__(self, img_input_dim=4096, img_output_dim=1024,
                 txt_input_dim=1024, txt_output_dim=1024, output_dim=10, minus_one_dim=1024):
        super(VGGIDCM_NN, self).__init__()
        self.img_net = VGGImgNN(img_input_dim, img_output_dim)
        self.txt_net = TextNN(txt_input_dim, txt_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

        self.img_layer = nn.Linear(minus_one_dim, minus_one_dim)
        self.txt_layer = nn.Linear(minus_one_dim, minus_one_dim)

    def forward(self, img, txt):
        view1_feature = self.img_net(img)
        view2_feature = self.txt_net(txt)

        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)

        stu_img_feature = F.relu(self.img_layer(view1_feature))
        stu_txt_feature = F.relu(self.txt_layer(view2_feature))

        return view1_feature, view2_feature, view1_predict, view2_predict, stu_img_feature, stu_txt_feature
