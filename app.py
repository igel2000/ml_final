# app.py
import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from colorize_model import ColorizationModel
import utils
import numpy as np

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = ColorizationModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()  # Переключаем в режим оценки

transform2 = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Функция обработки изображения
def colorize_image(input_image):
    # Подготовка изображения
    img = transform2(input_image).unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        output_ab = model(img)  # [1, 2, H, W]

    colorized_image = utils.LAB_to_RGB(img.cpu().detach(), output_ab.cpu().detach()).squeeze(0) #.permute(1, 2, 0)

    output_image =  transforms.ToPILImage()(colorized_image)
    return output_image

# Интерфейс
title = "Colorization Model 09"
description = "Загрузите черно-белое изображение, и приложение раскрасит его с помощью модели ColorizationModel09."

interface = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Черно-белое изображение"),
    outputs=gr.Image(type="pil", label="Раскрашенное изображение"),
    title=title,
    description=description,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)