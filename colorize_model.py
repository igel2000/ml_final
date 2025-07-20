# model.py

import torch
import torch.nn as nn
import torchvision.models as models
import timm

# Идея использовать InceptionResNetV2 взята отсюда: https://habr.com/ru/companies/nix/articles/342388/

class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        # Encoder: предобученная InceptionResNetV2, которая извлекает признаки
        self.encoder = timm.create_model('inception_resnet_v2', pretrained=True, features_only=True)

        # Пример выходных размеров: [80, 160, 192, 208, 1536]

        # Decoder: сверточная сеть, которая восстанавливает цветное изображение из этих признаков.
        self.decoder = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, padding=1), # уменьшает количество каналов.
            nn.ReLU(),                                      # нелинейная активация.
            nn.Upsample(scale_factor=2),                    # увеличивает разрешение изображения в 2 раза.
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # предсказывает 2 цветовых канала ab в цветовом пространстве Lab
            nn.Tanh(), # нормализация в диапазоне [-1, 1]
            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # inception_resnet_v2 ожидает три канала, а у нас 1. Размножим его
        x = x.repeat(1, 3, 1, 1)  # [B, 3, H, W]        
        features = self.encoder(x)
        x = self.decoder(features[-1])
        return x

    @staticmethod
    def model_name():
        return "ColorizationModel" if torch.cuda.is_available() else "ColorizationModel"    
