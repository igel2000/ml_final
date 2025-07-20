FROM ubuntu:24.04

# Установка необходимных зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    gcc \
    g++ \
    make \
    cmake \
    python3-dev \
    build-essential \
    python3.12-venv \
    libgl1 \
    libx11-6 \
    libglib2.0-0 \    
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаем и активируем виртуальное окружение
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip

# Установка gdown для скачивания с Google Drive
RUN pip install gdown

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 

# Установить зависимости
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Копируем код
COPY ./app.py .
COPY ./utils.py .
COPY ./colorize_model.py .

# Скачивание модели с Google Drive
# ID файла: 1Wimklwn0BiUM4noQqaOPrGrHxVMdAcP4
RUN gdown "https://drive.google.com/uc?id=1Wimklwn0BiUM4noQqaOPrGrHxVMdAcP4" -O model.pth

# Запуск приложения
CMD ["python", "app.py"]