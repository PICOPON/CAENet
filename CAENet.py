import torch.nn as nn


# 网络模型定义
class CAENet(nn.Module):
    def __init__(self):
        super(CAENet, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(8, 16, (3,3), (1,1), 1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, (3,3), (1,1), 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 3, (3,3), (1,1), 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(3, 8, (3,3), (1,1), 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, (3,3), (1,1), 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return self.decoder_cnn(x)


# 模型参数初始化
def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        # nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        # nn.init.constant_(layer.bias, 0.1)
        pass