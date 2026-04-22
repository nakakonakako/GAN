import torch
import torch.nn as nn

# エンコーダブロック(画像の圧縮と特徴抽出)
class UNetDown(nn.Module):
  def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
    super(UNetDown, self).__init__()
    layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
      layers.append(nn.BatchNorm2d(out_size))
    layers.append(nn.LeakyReLU(0.2))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.model(x)

# デコーダブロック (画像の拡大と特徴の復元)
class UNetUp(nn.Module):
  def __init__(self, in_size, out_size, dropout=0.0):
    super(UNetUp, self).__init__()
    layers = [
      nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(out_size),
      nn.ReLU(inplace=True)
    ]
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)

  def forward(self, x, skip_input):
    x = self.model(x)
    x = torch.cat((x, skip_input), 1) # スキップ接続で特徴を結合
    return x


class GeneratorUNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super().__init__()
    
    # エンコーダ
    self.down1 = UNetDown(in_channels, 64, normalize=False)
    self.down2 = UNetDown(64, 128)
    self.down3 = UNetDown(128, 256)
    
    # ボトルネック
    self.down4 = UNetDown(256, 512, normalize=False)
    
    # デコーダ スキップ結合で入力チャネル数が倍
    self.up1 = UNetUp(512, 256, dropout=0.5) 
    self.up2 = UNetUp(512, 128)
    self.up3 = UNetUp(256, 64)
    
    self.final = nn.Sequential(
      nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
    )
    
  def forward(self, x):
    # スキップ結合のために各層の出力を保持
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    
    u1 = self.up1(d4, d3) # ボトルネックとエンコーダの出力を結合
    u2 = self.up2(u1, d2)
    u3 = self.up3(u2, d1)
    
    return self.final(u3)
    