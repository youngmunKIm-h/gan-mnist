import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
# ... 나머지 import
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

# 하이퍼파라미터
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100      # 노이즈 벡터 차원
img_dim = 28 * 28     # MNIST 이미지 크기 (flatten)
batch_size = 128
lr = 2e-4
epochs = 50

# 데이터 로드 (픽셀값을 [-1, 1]로 정규화 → Generator 출력의 tanh와 매칭)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator: 노이즈 → 이미지
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh(),  # 출력 범위 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# Discriminator: 이미지 → 진짜/가짜 확률
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 진짜일 확률
        )

    def forward(self, img):
        return self.net(img)


G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

os.makedirs("samples", exist_ok=True)
fixed_noise = torch.randn(64, latent_dim, device=device)  # 학습 경과 확인용 고정 노이즈

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
        bs = real_imgs.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # --- Discriminator 학습 ---
        # 진짜는 1, 가짜는 0으로 판별하도록
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)

        loss_real = criterion(D(real_imgs), real_labels)
        loss_fake = criterion(D(fake_imgs.detach()), fake_labels)  # detach로 G 그래디언트 차단
        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Generator 학습 ---
        # D가 가짜를 진짜(1)로 착각하게 만들도록
        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        loss_G = criterion(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  loss_D: {loss_D.item():.4f}  loss_G: {loss_G.item():.4f}")

    # 매 에폭마다 샘플 이미지 저장
    with torch.no_grad():
        samples = G(fixed_noise).view(-1, 1, 28, 28)
        samples = (samples + 1) / 2  # [-1,1] → [0,1]
        save_image(samples, f"samples/epoch_{epoch+1:03d}.png", nrow=8)

print("학습 완료. samples/ 폴더에서 생성 이미지 확인 가능.")