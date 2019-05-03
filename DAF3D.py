from torch import nn
import torch
import torch.nn.functional as F
from BackBone3D import BackBone3D


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DAF3D(nn.Module):
    def __init__(self):
        super(DAF3D, self).__init__()
        self.backbone = BackBone3D()

        self.down4 = nn.Sequential(
            nn.Conv3d(2048, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(1024, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.attention4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.refine4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()

        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(),)

        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)

        self.predict1_4 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv3d(128, 1, kernel_size=1)

        self.predict2_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv3d(64, 1, kernel_size=1)

        self.predict = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        # Top-down
        down4 = self.down4(layer4)
        down3 = torch.add(
            F.upsample(down4, size=layer3.size()[2:], mode='trilinear'),
            self.down3(layer3)
        )
        down2 = torch.add(
            F.upsample(down3, size=layer2.size()[2:], mode='trilinear'),
            self.down2(layer2)
        )
        down1 = torch.add(
            F.upsample(down2, size=layer1.size()[2:], mode='trilinear'),
            self.down1(layer1)
        )
        down4 = F.upsample(down4, size=layer1.size()[2:], mode='trilinear')
        down3 = F.upsample(down3, size=layer1.size()[2:], mode='trilinear')
        down2 = F.upsample(down2, size=layer1.size()[2:], mode='trilinear')

        predict1_4 = self.predict1_4(down4)
        predict1_3 = self.predict1_3(down3)
        predict1_2 = self.predict1_2(down2)
        predict1_1 = self.predict1_1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))

        predict2_4 = self.predict2_4(refine4)
        predict2_3 = self.predict2_3(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict2_1 = self.predict2_1(refine1)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)

        aspp = self.aspp_gn(self.aspp_conv(aspp))

        predict = self.predict(aspp)

        predict1_1 = F.upsample(predict1_1, size=x.size()[2:], mode='trilinear')
        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='trilinear')
        predict1_3 = F.upsample(predict1_3, size=x.size()[2:], mode='trilinear')
        predict1_4 = F.upsample(predict1_4, size=x.size()[2:], mode='trilinear')

        predict2_1 = F.upsample(predict2_1, size=x.size()[2:], mode='trilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='trilinear')
        predict2_3 = F.upsample(predict2_3, size=x.size()[2:], mode='trilinear')
        predict2_4 = F.upsample(predict2_4, size=x.size()[2:], mode='trilinear')

        predict = F.upsample(predict, size=x.size()[2:], mode='trilinear')

        if self.training:
            return predict1_1, predict1_2, predict1_3, predict1_4, \
                   predict2_1, predict2_2, predict2_3, predict2_4, predict
        else:
            return predict
