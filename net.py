import torch.nn as nn
# import torch.nn.functional as F


class MACNet(nn.Module):
    def __init__(self, d_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.drop = nn.Dropout(d_prob)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16*136*4, 20)
        self.bn3 = nn.BatchNorm1d(num_features=20)

        self.fc2 = nn.Linear(20, 84)
        self.bn4 = nn.BatchNorm1d(num_features=84)

        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = out.view(out.shape[0], -1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.drop(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out

    # def __init__(self, classes):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.bn1 = nn.BatchNorm2d(num_features=6)
    #
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.bn2 = nn.BatchNorm2d(num_features=16)
    #
    #     self.fc1 = nn.Linear(16, 120)
    #     self.bn3 = nn.BatchNorm1d(num_features=120)
    #
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, classes)
    #
    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = F.relu(out)
    #
    #     out = F.max_pool2d(out, 2)
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = F.relu(out)
    #
    #     out = F.max_pool2d(out, 2)
    #
    #     out = out.view(-1, 16)
    #
    #     out = self.fc1(out)
    #     out = self.bn3(out)
    #     out = F.relu(out)
    #
    #     out = F.relu(self.fc2(out))
    #     out = self.fc3(out)
    #     return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
