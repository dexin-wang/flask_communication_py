import torch
from torch import nn
from torch.nn import functional as F


class ResBlock2(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()

        stride_1 = 1 if ch_out == ch_in else 2

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride_1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out),
                # nn.ReLU()
            )
        else:
            self.extra = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):
    def __init__(self, cls=1000):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_1 = ResBlock2(64, 64)
        self.conv2_2 = ResBlock2(64, 64)
        self.conv3_1 = ResBlock2(64, 128)
        self.conv3_2 = ResBlock2(128, 128)
        self.conv4_1 = ResBlock2(128, 256)
        self.conv4_2 = ResBlock2(256, 256)
        self.conv5_1 = ResBlock2(256, 512)
        self.conv5_2 = ResBlock2(512, 512)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, cls)

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out3 = self.conv3_1(out2)
        out3 = self.conv3_2(out3)
        out4 = self.conv4_1(out3)
        out4 = self.conv4_2(out4)
        out5 = self.conv5_1(out4)
        out5 = self.conv5_2(out5)
        out = self.avgpool(out5)
        out = self.fc(out.view(out.size(0), -1))    # 先展平，再进全连接层

        return [out1, out2, out3, out4, out5, out]

def compute_loss(net, xc, yc):

    y_pred = net(xc)[-1]

    loss = F.cross_entropy(y_pred, yc.long())

    return loss