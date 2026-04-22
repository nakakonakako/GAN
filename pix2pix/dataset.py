import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FacadesDataset(Dataset):
  def __init__(self, root_dir="./data", mode="train"):
    self.img_dir = os.path.join(root_dir, mode)
    self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
    
    # Pix2Pixの標準である256x256にリサイズするための変換
    self.transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]に正規化
    ])
    
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    img_name = os.path.join(self.img_dir, self.image_files[idx])
    img = Image.open(img_name).convert("RGB")
    
    # 画像は横に2枚連結されている
    w, h = img.size
    half_w = w // 2
    
    # 左半分が実際の写真、右半分がラベル画像
    real_B_pil = img.crop((0, 0, half_w, h))  # 実際の写真
    real_A_pil = img.crop((half_w, 0, w, h))  # ラベル画像
    
    real_A = self.transform(real_A_pil)
    real_B = self.transform(real_B_pil)
    
    return real_A, real_B