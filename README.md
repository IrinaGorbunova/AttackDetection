# Attack Detection

## Задача:

Реализовать модель классификации видео. Все анализируемые видео - короткие (порядка 2-3-х секунд), содержение - селфи.
Есть два класса видео:
1) нормальные селфи видео ("оригиналы") в которых в кадре один человек, 
2) селфи видео, в части кадров которого лицо снимаемого закрыто фотографией или другим изображением другого человека ("комбинированная атака").

## Данные:

- Основной датасет [COMBINED DS](https://drive.google.com/file/d/1SB0qwhhlEFH1DZNeaFrsGEbYfWerRxWc/view?usp=sharing)
- Дополнительный датасет [CASIA](https://drive.google.com/file/d/186x9PV_8jVD8cj0gC-s-rawH0Q-U9bfI/view?usp=sharing)\
Датасет CASIA содержит три вида атак: распечатанное фото, распечатанное фото с вырезанными глазами и показ видео на другом устройстве. В нем нет комбинированных атак в чистом виде, но каждый из присутствующих в нем видов атак может выступать как часть комбинированной атаки (при комбинированной атаке на части кадров лицо снимаемого может быть закрыто, например, распечатанным фото, распечатанным фото с вырезанными глазами или фото/видео, показываемым на другом устройстве). Поэтому добавление его к обучающему датасету может повысить качество распознавания комбинированных атак (что мы и наблюдаем).

## Решение:

В данном репозитории представлено две модели для классификации видео: CNN-Transformer и CNN-RNN.

CNN извлекает информацию из каждого кадра по отдельности (формирует эмбеддинги). CNN часть в обеих моделях представляет собой ResNet 34 без последнего fc слоя. В CNN-Transformer все слои ResNet участвуют в обучении, а в CNN-RNN большая часть слоев (кроме последнего сверточного блока) заморожена.

RNN и трансформер обрабатывают всю последовательность кадров (полученных эмбеддингов), за счет чего способны улавливать происходящие при комбинированных атаках изменения (например, смену настоящего лица на одном кадре на фотографию на другом кадре). RNN часть представляет собой один слой LSTM, а трансформер состоит из одного слоя TransformerEncoder.  

### CNN-Transformer

В папке CNN_Transformer можно найти схему с [архитектурой](https://github.com/IrinaGorbunova/AttackDetection/blob/main/CNN_Transformer/CNN-Transformer.png) модели, а также два ноутбука:
- [v1_CNN_Transformer.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/CNN_Transformer/v1_CNN_Transformer.ipynb) - обучение модели только на основном датасете.
- [v2_CNN_Transformer.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/CNN_Transformer/v2_CNN_Transformer.ipynb) - обучение модели на основном и дополнительном датасетах.

### CNN-RNN

В папке CNN_RNN можно найти схему с [архитектурой](https://github.com/IrinaGorbunova/AttackDetection/blob/main/CNN_RNN/CNN_RNN.png) модели, а также ноутбук
[v2_CNN_RNN.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/CNN_RNN/v2_CNN_RNN.ipynb) c обучением модели на основном и дополнительном датасетах.

Сохраненные веса всех моделей можно скачать по [ссылке](https://drive.google.com/drive/folders/1CkgYQdyi9ZVBIDCiqMcrHak1L4ylgmRR?usp=sharing)

Ноутбук [Test.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/Test.ipynb) демонстрирует процесс получения предсказаний с помощью моделей.

Папка Examples содержит четыре примера видео для проведения теста.

В папке RNN_Autoencoder дополнительно представлена модель RNN автоэнкодера. Основной датасет содержит очень большое количество реальных примеров, что можно использовать для обучения автоэнкодера. Ошибка реконструкции на атаках будет выше, чем на реальных примерах, что можно использовать для их обнаружения. В папке лежат два ноутбука:
- [v1_RNN_AE.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/RNN_Autoencoder/v1_RNN_AE.ipynb) - для обучения используются эмбеддинги из ResNet 34,
- [v2_RNN_AE.ipynb](https://github.com/IrinaGorbunova/AttackDetection/blob/main/RNN_Autoencoder/v2_RNN_AE.ipynb) - для обучения используются эмбеддинги из CNN части модели v2_CNN_Transformer

и веса соответствующих моделей.
