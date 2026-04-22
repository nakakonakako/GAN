import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import FacadesDataset
from models.discriminator import Discriminator
from models.generator import GeneratorUNet

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs("images", exist_ok=True)
  
  batch_size = 4
  epochs = 10
  lr = 0.0002
  lambda_pixel = 100
  
  dataset = FacadesDataset(mode="train")
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  # モデルの初期化
  generator = GeneratorUNet(in_channels=3, out_channels=3).to(device)
  discriminator = Discriminator(in_channels=3).to(device)
  
  criterion_GAN = nn.BCEWithLogitsLoss() # GANの損失関数
  criterion_pixelwise = nn.L1Loss() # ピクセル単位の損失関数
  
  optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
  optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
  
  print("Training started...")
  
  for epoch in range(epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
      real_A = real_A.to(device) # ラベル画像
      real_B = real_B.to(device) # 実際の写真
      
      # PatchGANの出力サイズに合わせた正解ラベルの定義 (入力64*64の場合、出力は3*3)
      valid = torch.ones((real_A.size(0), 1, 3, 3), requires_grad=False).to(device)
      fake = torch.zeros((real_A.size(0), 1, 3, 3), requires_grad=False).to(device)
      
      
      """生成器の学習"""
      optimizer_G.zero_grad()
      fake_B = generator(real_A)
      
      # 識別器を騙すための損失
      pred_fake = discriminator(real_A, fake_B)
      loss_GAN = criterion_GAN(pred_fake, valid)
      
      # 正解画像との形状の一致を示す損失
      loss_pixel = criterion_pixelwise(fake_B, real_B)
      
      # 2つの損失の合算
      loss_G = loss_GAN + lambda_pixel * loss_pixel
      loss_G.backward()
      optimizer_G.step()
  
      
      """識別器の学習"""
      optimizer_D.zero_grad()

      # 本物ペアの評価
      pred_real = discriminator(real_A, real_B)
      loss_real = criterion_GAN(pred_real, valid)
      
      # 偽物ペアの評価
      pred_fake = discriminator(real_A, fake_B.detach())
      loss_fake = criterion_GAN(pred_fake, fake)
      
      # 識別器の損失
      loss_D = 0.5 * (loss_real + loss_fake)
      loss_D.bachward()
      optimizer_D.step()
      
    print(f"[Epoch {epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
    save_sample(generator, real_A, real_B, epoch + 1, device)
    
def save_sample(generator, real_A, real_B, epoch, device):
  """生成画像を保存する関数"""
  generator.eval()
  with torch.no_grad():
    fake_B = generator(real_A[:4])
  generator.train()
  
  # 描画のために[-1, 1]から[0, 1]に変換
  real_A = (real_A[:4].cpu() + 1) / 2
  real_B = (real_B[:4].cpu() + 1) / 2
  fake_B = (fake_B.cpu() + 1) / 2
  
  fig, axes = plt.subplots(3, 4, figsize=(8, 6))
  for j in range(4):
    axes[0, j].imshow(real_A[j].squeeze(), cmap="gray")
    axes[0, j].axis("off")
    if j == 0: axes[0, j].set_title("Input (Masked)")
    
    axes[1, j].imshow(fake_B[j].squeeze(), cmap="gray")
    axes[1, j].axis("off")
    if j == 0: axes[1, j].set_title("Generated")
    
    axes[2, j].imshow(real_B[j].squeeze(), cmap="gray")
    axes[2, j].axis("off")
    if j == 0: axes[2, j].set_title("Target (Real)")
    
  plt.tight_layout()
  plt.savefig(f"images/epoch_{epoch}.png")
  plt.close()
    
if __name__ == "__main__":
  train()