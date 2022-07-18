import torch
import torch.nn as nn


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, out_channel=1):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(
            ndf * 8, out_channel, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class ASPP(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super().__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

# 这是在对抗像素级的特征，这是否可以使用在CIDA中。
class FeatDiscriminator(nn.Module):
    def __init__(self, inplanes=2048, dilation_series=[6, 12, 18, 24], padding_series=[6, 12, 18, 24], ndf=128):
        super().__init__()
        self.layer1 = ASPP(inplanes, dilation_series, padding_series, ndf)
        self.layer2 = ASPP(ndf, dilation_series,
                           padding_series, int(ndf / 2))
        self.layer3 = nn.Conv2d(
            int(ndf / 2), int(ndf / 4), kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(int(ndf / 4), 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.layer3(x)
        x = self.leaky_relu(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    model = FeatDiscriminator()
    x = torch.randn(1, 1024, 65, 129)
    y = model(x)
    print(y.shape)
