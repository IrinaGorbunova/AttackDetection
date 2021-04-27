import torch
import argparse
from models import CNN_Transformer, CNN_RNN
from preprocess import get_frames, transform_frames


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_path, mode='transformer'):

    if mode == 'transformer':
        model = CNN_Transformer(num_classes=2,
                                nlayers=1,
                                hidden=32,
                                nhead=4,
                                dim_feedforward=128,
                                dropout=0.5)
    else:
        model = CNN_RNN()

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print('Wrong mode or model path!')

    return model

 
def prepare_input(filename, n_max=float('inf')):

    frames, f_len = get_frames(filename, n_max=n_max)
    img_tensor = transform_frames(frames)

    return img_tensor.unsqueeze(0), torch.LongTensor([f_len])


def predict(model, mode, img, img_len):

    model = model.to(device)
    model.eval()

    if mode == 'transformer':
        output = model(img.to(device), img_len, mask=None)
    else:
        output = model(img.to(device), img_len)

    return torch.softmax(output, -1).max(-1)[0].detach().tolist()[0], \
            output.max(-1)[1].detach().tolist()[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default='transformer', help=\
        "Тип модели (rnn, transformer)")
    parser.add_argument("-p", "--path", default='', help="Путь к весам модели")
    parser.add_argument("-f", "--filename", default='', help="Путь к файлу с видео")
    parser.add_argument("-n", "--n", default=float('inf'), help="Количество кадров")
    
    args = parser.parse_args()

    filename = args.filename
    mode = args.mode
    model_path = args.path
    n_max = args.n

    model = get_model(model_path, mode)
    img, img_len = prepare_input(filename, n_max)
    prob, out = predict(model, mode, img, img_len)

    if out == 1:
        print('Attack : {:.2%}'.format(prob))
    else:
        print('Original : {:.2%}'.format(prob))