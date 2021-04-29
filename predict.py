import torch
import argparse
from models import CNN_Transformer, CNN_RNN
from preprocess import get_frames, transform_frames


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_path, mode='transformer'):
    """
    Возвращает модель указанного типа с загруженными весами

    params:
        model_path - строка, путь к файлу с сохраненными весами модели
        mode - строка, тип модели 'rnn' или 'transformer' (по умолчанию)
    """

    if mode == 'transformer':
        model = CNN_Transformer(num_classes=2,
                                nlayers=1,
                                hidden=32,
                                nhead=4,
                                dim_feedforward=128,
                                dropout=0.5)
    else:
        model = CNN_RNN(num_classes=2, 
                        rnn_num_layers=1, 
                        hidden=64, 
                        rnn_hidden_size=64, 
                        bidirectional=False, 
                        dropout=0.)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print('Неверно указан тип модели или путь к весам!')

    return model

 
def prepare_input(filename, n_max=float('inf')):
    """
    Разбивает видео на кадры и преобразует в тензор

    params:
        filename - строка, путь к видеофайлу
        n_max - число, количество кадров для анализа. 
                По умолчанию float('inf'), т.е. берутся все кадры видео.
    """

    frames, f_len = get_frames(filename, n_max=n_max)
    img_tensor = transform_frames(frames)

    return img_tensor.unsqueeze(0), torch.LongTensor([f_len])


def predict(mode, model, imgs, imgs_len):
    """
    Для последовательности изображений 
    возвращает предсказанный моделью класс и его вероятность

    params:
        mode - строка, тип модели 'rnn' или 'transformer'
        model - модель CNN_RNN или CNN-Transformer
        imgs - тензор, последовательность кадров
        imgs_len - тензор, количество кадров в последовательности
    """

    model = model.to(device)
    model.eval()

    if mode == 'transformer':
        output = model(imgs.to(device))
    else:
        output = model(imgs.to(device), imgs_len)

    output = torch.softmax(output, -1).max(-1)

    return output[1].item(), output[0].item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default='transformer', 
                        help="Тип модели (rnn, transformer)")
    parser.add_argument("-w", "--weights", default='', help="Путь к весам модели")
    parser.add_argument("-f", "--file", default='', help="Путь к видео")
    parser.add_argument("-n", "--n", 
        help="Количество кадров для анализа. Если не указано, то берутся все.")
    
    args = parser.parse_args()
    
    if args.filename:
        filename = args.filename
    if args.mode:
        mode = args.mode
    if args.weights:
        model_path = args.weights
    if args.n:
        n_max = int(args.n)
    else:
        n_max = float('inf')

    model = get_model(model_path, mode)
    imgs, imgs_len = prepare_input(filename, n_max)
    out, prob = predict(mode, model, imgs, imgs_len)

    if out == 1:
        print('Attack : {:.2%}'.format(prob))
    else:
        print('Original : {:.2%}'.format(prob))