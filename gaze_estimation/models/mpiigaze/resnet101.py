import torch
import torch.nn as nn
import torchvision
import yacs.config

class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super(Model, self).__init__()
        
        resnet101 = torchvision.models.resnet101(pretrained=True)
        
        self.convNet = resnet101

        self.FC = nn.Sequential(
            nn.Linear(1000, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.output = nn.Sequential(
            nn.Linear(4096+2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )
        
        # input channels = 1
        self.convNet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

      

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.convNet(x)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self.FC(x)

        x = torch.cat([x, y], dim=1)
        x = self.output(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)