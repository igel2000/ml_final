import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

#from PIL import Image
from torchvision.transforms import ToTensor #, ToPILImage
#import kornia.color as kcolor
import kornia as K
#import kornia.utils as Kutils


def set_seed(seed):
    """Установить seed для всего"""
    # Базовые библиотеки
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # если используются GPU
    
    # Для воспроизводимости на CUDA (может снижать производительность!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Фиксируем seed для других операций (например, DataLoader)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def model_filename(model, models_folder, suffix=None):
    """Сформироватиь имя файла с сохраненными весами модели"""
    suffix = "" if suffix is None else f"_{suffix}"
    return Path(models_folder, f"{model.model_name()}{suffix}.pth")

def show_batch(loader, num_samples=4):
    batch = next(iter(loader))
    grey_batch, color_batch = batch

    if num_samples < 2:
        raise ValueError("Значение num_samples должно быть больше 1.")

    if num_samples < 6:
        fig, axes = plt.subplots(2, num_samples,  figsize=(2 * num_samples, 4 ))
    else:
        fig, axes = plt.subplots(num_samples, 2, figsize=(4, 2 * num_samples))

    for i in range(num_samples):
        def tensor_to_image(tensor):
            img = tensor[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            return img.clip(0, 1)
        grey_img = tensor_to_image(grey_batch)
        color_img = tensor_to_image(color_batch)

        if num_samples <= 5:                
            xg, yg = (0, i)
            xc, yc = (1, i)
        else:
            xg, yg = (i, 0)
            xc, yc = (i, 1)

        axes[xg][yg].imshow(grey_img, cmap='gray')
        axes[xg][yg].set_title("Ч/Б")
        axes[xg][yg].axis("off")

        axes[xc][yc].imshow(color_img)
        axes[xc][yc].set_title("Цветное")
        axes[xc][yc].axis("off")

    plt.tight_layout()
    plt.show()


def lab_to_rgb(L, ab):
    """
    Преобразует Lab изображение в RGB
    L: (batch, 1, H, W) в диапазоне [-1, 1] (0-100 в Lab)
    ab: (batch, 2, H, W) в диапазоне [-1, 1]
    """
    L = (L + 1.0) * 50.0  # [-1,1] -> [0, 100]
    ab = (ab + 1.0) * 128.0  # [-1,1] -> [0, 255]
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)

    rgb_images = []
    for img in Lab:
        img = img.astype(np.float32)
        img[..., 0] = np.clip(img[..., 0], 0, 100)
        img[..., 1] = np.clip(img[..., 1], -127, 128)
        img[..., 2] = np.clip(img[..., 2], -127, 128)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        rgb_images.append(img_rgb)
    return np.array(rgb_images)


def tensor_to_image(tensor):
    """Преобразует батч изображений в numpy-изображения для отображения."""
    images = tensor.permute(0, 2, 3, 1).cpu().numpy()
    images = (images + 1) / 2  # [-1,1] -> [0,1]
    return np.clip(images, 0, 1)


def show_batch_ab(loader, num_samples=4):
    batch = next(iter(loader))
    grey_batch, ab_batch = batch  # grey_batch: (B, 1, H, W), ab_batch: (B, 2, H, W)

    if num_samples < 2:
        raise ValueError("Значение num_samples должно быть больше 1.")

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

    # Конвертируем L + ab в RGB
    color_batch = lab_to_rgb(grey_batch, ab_batch)

    for i in range(num_samples):
        # Получаем отдельные изображения
        grey_img = tensor_to_image(grey_batch)[i]  # (H, W, 1)
        color_img = color_batch[i]  # (H, W, 3)

        # Отображаем
        axes[i, 0].imshow(grey_img.squeeze(), cmap='gray')
        axes[i, 0].set_title("L-канал (Ч/Б)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(color_img)
        axes[i, 1].set_title("Цветное (Lab → RGB)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def RGB_to_LAB(image_pil):
    """
    Преобразует Pil.Image в пространстве RGB в LAB-пространство,
    возвращает нормализованные тензоры L и ab в диапазоне [-1, 1].
    """
    # Преобразование в тензор (HxWx3, диапазон [0,1])
    image_tensor = ToTensor()(image_pil)  # [0,1] range

    # Добавление batch-размерности и преобразование RGB -> LAB
    image_rgb = image_tensor.unsqueeze(0)  # shape: (1, 3, H, W)
    image_lab = K.color.rgb_to_lab(image_rgb)  # LAB в диапазоне L: [0, 100], a/b: [-128, 127]

    # Разделение каналов
    L = image_lab[:, 0:1, :, :]  # shape: (1, 1, H, W)
    ab = image_lab[:, 1:3, :, :]  # shape: (1, 2, H, W)

    # Нормализация L: [0, 100] -> [-1, 1]
    L = (L - 50.0) / 50.0

    # Нормализация ab: [-128, 127] -> [-1, 1]
    ab = ab / 128.0

    return L.squeeze(0), ab.squeeze(0)  # Убираем лишнее измерение


def RGBTensor_to_LAB(image_tensor):
    """
    Преобразует Pil.Image в пространстве RGB в LAB-пространство,
    возвращает нормализованные тензоры L и ab в диапазоне [-1, 1].
    """
    # Преобразование в тензор (HxWx3, диапазон [0,1])
    #image_tensor = ToTensor()(image_pil)  # [0,1] range

    # Добавление batch-размерности и преобразование RGB -> LAB
#    print(f'image_tensor.shape: {image_tensor.shape}, type: {type(image_tensor)}')
    image_rgb = image_tensor.unsqueeze(0)  # shape: (1, 3, H, W)
    image_lab = K.color.rgb_to_lab(image_rgb)  # LAB в диапазоне L: [0, 100], a/b: [-128, 127]

    # Разделение каналов
    L = image_lab[:, 0:1, :, :]  # shape: (1, 1, H, W)
    ab = image_lab[:, 1:3, :, :]  # shape: (1, 2, H, W)

    # Нормализация L: [0, 100] -> [-1, 1]
    L = (L - 50.0) / 50.0

    # Нормализация ab: [-128, 127] -> [-1, 1]
    ab = ab / 128.0

    return L.squeeze(0), ab.squeeze(0)  # Убираем лишнее измерение

def LAB_to_RGB(L, ab):
    """
    Принимает нормализованные тензоры L и ab в диапазоне [-1, 1],
    денормализует их, преобразует LAB -> RGB и сохраняет как JPEG.
    """
    # Денормализация L: [-1, 1] -> [0, 100]
    L = L * 50.0 + 50.0

    # Денормализация ab: [-1, 1] -> [-128, 127]
    ab = ab * 128.0

    # Объединение каналов LAB
    image_lab = torch.cat([L, ab], dim=1)  # shape: (1, 3, H, W)

    # Преобразование LAB -> RGB
    image_rgb = K.color.lab_to_rgb(image_lab)  # диапазон [0, 1]

    return image_rgb    