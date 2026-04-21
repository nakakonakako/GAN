import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 50
LATENT_DIM = 100 # 生成器に入力するノイズの次元数
IMAGE_DIM = 28 * 28 # MNIST 画像サイズ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(LATENT_DIM, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, IMAGE_DIM),
      nn.Tanh() # 出力を[-1, 1]の範囲にするためにTanhを使用
    )
    
  def forward(self, z):
    img = self.model(z)
    return img
  
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(IMAGE_DIM, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 1),
    )
    
  def forward(self, img):
    validity = self.model(img)
    return validity

# モデルのインスタンス化
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 損失関数と最適化手法
criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# 画像のピクセル値を[-1, 1]の範囲に正規化するための変換
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

os.makedirs("images", exist_ok=True)

for epoch in range(EPOCHS):
  for i, (imgs, _) in enumerate(dataloader):
    
    real_imgs = imgs.view(imgs.size(0), -1).to(device)
    real_labels = torch.ones(imgs.size(0), 1).to(device)
    fake_labels = torch.zeros(imgs.size(0), 1).to(device)
    

    optimizer_D.zero_grad()
    
    # 本物画像の損失を計算
    outputs_real = discriminator(real_imgs)
    d_loss_real = criterion(outputs_real, real_labels)
    
    # 偽物画像(生成画像)の損失を計算
    z = torch.randn(imgs.size(0), LATENT_DIM).to(device)
    fake_imgs = generator(z)
    outputs_fake = discriminator(fake_imgs.detach())
    d_loss_fake = criterion(outputs_fake, fake_labels)
    
    # 識別機の全体の損失とパラメータ更新
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()
    
    optimizer_G.zero_grad()
    
    outputs = discriminator(fake_imgs)
    g_loss = criterion(outputs, real_labels)
    
    g_loss.backward()
    optimizer_G.step()
    
  print(f"[Epoch {epoch+1}/{EPOCHS}] [D_loss: {d_loss.item():.4f}] [G_loss: {g_loss.item():.4f}]")
  
  if (epoch + 1) % 10 == 0:
    with torch.no_grad():
      sample_z = torch.randn(64, LATENT_DIM).to(device)
      sample_imgs = generator(sample_z).view(-1, 1, 28, 28)
      
      fig, axes = plt.subplots(4, 4, figsize=(4, 4))
      for j, ax in enumerate(axes.flatten()):
        ax.imshow(sample_imgs[j].cpu().squeeze(), cmap="gray")
        ax.axis("off")
      
      plt.savefig(f"images/epoch_{epoch+1}.png")
      plt.close()
    