import cv2
import ffmpeg
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


h, w = 224, 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def check_rotation(path_video_file):
    "Возвращает код для корректного поворота видео"

    meta_dict = ffmpeg.probe(path_video_file)

    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except: 
        pass
        
    return rotateCode


def correct_rotation(frame, rotateCode):
    "Выполняет поворот кадра в соответствии с указанным кодом"

    return cv2.rotate(frame, rotateCode)


def get_frames(filename, n_max=float('inf')):
    "Разбивает видео на указанное количество кадров"
    
    frames = []
    v_cap = cv2.VideoCapture(filename)
    rotateCode = check_rotation(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    n_frames = min(v_len, n_max)
    frame_list= np.linspace(0, v_len-1, n_frames, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)  
            frames.append(frame)

    v_cap.release()

    return frames, len(frames)


def transform_frames(frames):
    "Преобразует список кадров в тензор"

    img_transform = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])    
    frames_tr = []

    for frame in frames:
        frame = Image.fromarray(frame)       
        frame_tr = img_transform(frame)
        frames_tr.append(frame_tr)
        
    imgs_tensor = torch.stack(frames_tr)    

    return imgs_tensor