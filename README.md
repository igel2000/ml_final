# Раскрашивания черно-белых изображений

Обучение модели для раскрашивания черно-белых изображений и приложения на базе gradio для демонстарции работы модели.

# Технологии
Python
OpenCv
PyTorch
Gradio
Docker


# Подход к решению задачи
Идея к решению задачи взята из статьи https://habr.com/ru/companies/nix/articles/342388/.

Для раскрашивания используется модель состоящая из:
  * энкодера, в качестве которого используется  [Inception Resnet V2](https://research.google/blog/improving-inception-and-image-classification-in-tensorflow/)
  * декодера - сверточной сети из серии блоков Conv2d+ReLU+Upsample

Для обучения использовались сразу два датасета:
  * [theblackmamba31/landscape-image-colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)
  * [aayush9753/image-colorization-dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset)
Из этих датасетов собирается микс, который делится на тренировочный и валидационный набор - так результаты обучения получаются лучше.

Приложение, демонстрирующее работы, реализовано с использованием gradio и docker.

# Обучение модели
Для обучения модели необходимо выполнить jupyter-ноутбук - train_model.ipynb.
В ноутбуке есть возможность включить отладочный режим, в котором для обучения используется только подмножетсво датасета и обучение происходит на меньшем количестве эпох.

В ходе обучения сохраняются:
  * веса модели после каждой эпохи
  * веса лучшей модели с минимальной loss по тренировочному набору 
  * веса лучшей модели с минимальной loss по валидационному набору (предполагается, что эти веса дают лучший результат)

В ходе обучения можно проверять качество полученных весов используя ноутбук inference_model.ipynb

# Приложение

Приложение для демонстрации раскрашивания картинок рассчитано, в первую очередь для работы в docker.

Собрать docker image
```
docker build -t colorization-app .
```
При построении образа веса модели будут скачаны по ссылке https://drive.google.com/uc?id=1Wimklwn0BiUM4noQqaOPrGrHxVMdAcP4 и сохранены в образе.

Запустить docker container
```
docker run -p 7860:7860 --rm colorization-app
```

Приложение будет доступно по адресу [http://localhost:7860/]

# Содержимое репозитория

Общие файлы:
* colorize_model.py - код модели
* utils.py - вспомогательные функции\
* settings - файл настроек

Обучение модели:
* train_model.ipynb - основной ноутбук для обучения модели
* inference_model.ipynb - ноубук для оперативной проверки моделей

Приложение:
* app.py - код приложения
* Dockerfile - описание образа приложения
* requirements_app.txt - requirements для приложения

# Источники
* [Раскрашиваем чёрно-белую фотографию с помощью нейросети из 100 строк кода](https://habr.com/ru/companies/nix/articles/342388/)
* [Реставрируем фотографии с помощью нейросетей](https://habr.com/ru/companies/vk/articles/453872/)
* [GAN-colorize-images](https://www.kaggle.com/code/yuriipolulikh/gan-colorize-images)
* [Image Colorization Using GANs](https://www.kaggle.com/code/ziyadelshazly/image-colorization-using-gans#Fitting-with-good-results)
* https://github.com/IlyaKuprik/ColorizeImage
* https://github.com/jantic/DeOldify/tree/master?tab=readme-ov-file#why-three-models
* [О цветовых пространствах](https://habr.com/ru/articles/181580/)