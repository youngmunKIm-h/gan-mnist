# Simple GAN for MNIST

PyTorch로 구현한 간단한 GAN(Generative Adversarial Network)입니다. MNIST 손글씨 숫자 이미지를 생성합니다. Fully-connected 레이어 기반의 최소 구조로, GAN의 학습 원리를 이해하는 데 초점을 맞췄습니다.

## 구조

두 개의 네트워크가 적대적으로 경쟁하며 학습합니다.

- **Generator**: 100차원 노이즈 벡터를 입력받아 28×28 이미지를 생성. `Linear → LeakyReLU` 스택에 출력은 `Tanh`([-1, 1] 범위).
- **Discriminator**: 이미지를 입력받아 진짜/가짜 확률을 출력. `Linear → LeakyReLU → Dropout` 스택에 출력은 `Sigmoid`.

두 네트워크는 미니맥스 게임을 벌입니다. Generator는 Discriminator를 속이도록, Discriminator는 진짜와 가짜를 구분하도록 번갈아 업데이트됩니다.

## 요구 사항

설치:

```bash
pip install torch torchvision
```

## 실행

```bash
python gan.py
```

- MNIST 데이터셋은 최초 실행 시 `./data`에 자동으로 다운로드됩니다.
- 매 에폭마다 생성 샘플이 `./samples/epoch_XXX.png`로 저장됩니다.
- CPU에서도 동작하지만 GPU(CUDA) 환경을 권장합니다.

## 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| latent_dim (노이즈 차원) | 100 |
| batch_size | 128 |
| learning rate | 2e-4 |
| optimizer | Adam (betas=0.5, 0.999) |
| epochs | 50 |

데이터는 `[-1, 1]`로 정규화되며, 이는 Generator 출력의 `Tanh` 범위와 일치시키기 위한 설정입니다.

## Loss 해석

학습이 정상적으로 진행되면 다음과 같은 궤적을 보입니다.

- **초반**: `loss_D`가 낮고(~0.4) `loss_G`가 높음(~2.5). Discriminator가 미숙한 Generator를 쉽게 구분하는 단계.
- **후반**: `loss_D` ≈ 1.0, `loss_G` ≈ 1.2 부근에서 안정화. Discriminator가 진짜/가짜를 절반 정도밖에 구분하지 못하는 상태로, GAN이 도달해야 할 균형점(Nash equilibrium)에 가깝습니다. `-log(0.5) ≈ 0.69`을 기준으로 두 loss가 팽팽하게 맞선 형태가 이상적입니다.

## 한계

Fully-connected 구조라 생성된 숫자가 다소 흐릿하고 뭉개집니다. 형태는 알아볼 수 있는 수준입니다. 품질을 높이려면 Convolution 기반의 **DCGAN**으로 전환하는 것이 다음 단계입니다.

## 라이선스

MIT
