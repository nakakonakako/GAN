import torch
import torch.nn as nn

class DiscriminatorPatchGAN(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    
    def discriminator_block(in_filters, out_filters, normalization=True):
      layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
      if normalization:
        layers.append(nn.BatchNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(
      # 入力は画像と条件画像をチャネル方向に結合したもの
      *discriminator_block(in_channels * 2, 64, normalization=False),
      *discriminator_block(64, 128),
      *discriminator_block(128, 256),
      *discriminator_block(256, 512),
      
      nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 最終的な出力は1チャネルの特徴マップ
    )
    
  def forward(self, img_A, img_B):
    # 入力画像と条件画像をチャネル方向に結合
    img_input = torch.cat((img_A, img_B), 1)
    return self.model(img_input)