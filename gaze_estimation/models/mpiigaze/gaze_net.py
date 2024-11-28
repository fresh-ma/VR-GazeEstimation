import torch
import torch.nn as nn
import torchvision
import yacs.config

class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super(Model, self).__init__()
        
        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.convNet = vgg16.features

        self.FC = nn.Sequential(
            nn.Linear(512*4*7, 4096),
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
        self.convNet[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # replace the maxpooling layer in VGG
        self.convNet[4] = nn.MaxPool2d(kernel_size=2, stride=1)
        self.convNet[9] = nn.MaxPool2d(kernel_size=2, stride=1)
      

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.convNet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.FC(x)

        x = torch.cat([x, y], dim=1)
        x = self.output(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

