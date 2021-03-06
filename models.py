import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNEncoder(nn.Module):
    "CNN энкодер на основе ResNet 34"

    def __init__(self, pretrained=True):
        super(CNNEncoder, self).__init__()
        
        resnet = models.resnet34(pretrained=pretrained) 
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, image):

        out = self.resnet(image)

        return out


class CNN_RNN(nn.Module):
    "Модель классификатора с архитектурой CNN-RNN"

    def __init__(self, num_classes, rnn_num_layers, hidden, rnn_hidden_size, bidirectional, dropout=0):

        super(CNN_RNN, self).__init__()

        self.hidden = hidden

        # CNN
        self.cnn = CNNEncoder()
        self.conv = nn.Conv2d(512, self.hidden, 1)
        
        # RNN
        if rnn_num_layers > 1:
            self.rnn = nn.LSTM(self.hidden, rnn_hidden_size, rnn_num_layers, 
                               bidirectional=bidirectional, dropout=dropout)
        else:
            self.rnn = nn.LSTM(self.hidden, rnn_hidden_size, rnn_num_layers, 
                               bidirectional=bidirectional)
        
        # Classifier
        self.dropout= nn.Dropout(dropout)
        if bidirectional:
            self.fc = nn.Linear(rnn_hidden_size*2, num_classes)
        else:
            self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, frames, f_lens):

        bs, s, c, height, width = frames.shape
        frames_emb = torch.zeros(s, bs, self.hidden).to(device)

        for i in range(s):
            img_emb = self.cnn(frames[:, i])
            img_emb = torch.relu(self.conv(img_emb))
            frames_emb[i] = img_emb.view(bs, -1)

        packed_frames_emb = nn.utils.rnn.pack_padded_sequence(frames_emb, f_lens, enforce_sorted=False)
        packed_out, h = self.rnn(packed_frames_emb)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.fc(self.dropout(torch.mean(out, 0)))
 
        return out


class PositionalEncoding(nn.Module):
    "Позиционное кодирование для трансформера"
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.scale * self.pe[:x.size(0), :]

        return self.dropout(x)


class CNN_Transformer(nn.Module):
    "Модель классификатора с архитектурой CNN-Transformer"

    def __init__(self, num_classes, nlayers, hidden, nhead, dim_feedforward, 
                 dropout=0, activation='relu'):

        super(CNN_Transformer, self).__init__()

        self.hidden = hidden
        
        # CNN
        self.cnn = CNNEncoder()
        self.conv = nn.Conv2d(512, hidden, 1)

        # Transformer
        self.pos = PositionalEncoding(hidden) 
        encoder_layer = nn.TransformerEncoderLayer(hidden, 
                                                   nhead, 
                                                   dim_feedforward, 
                                                   dropout, 
                                                   activation)
        encoder_norm = nn.LayerNorm(hidden)
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 nlayers, 
                                                 encoder_norm)

        # Classifier
        self.dropout= nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, frames, mask=None):
        
        bs, s, c, height, width = frames.shape
        frames_emb = torch.zeros(s, bs, self.hidden).to(device)

        for i in range(s):
            img_emb = self.cnn(frames[:, i])
            img_emb = torch.relu(self.conv(img_emb))
            frames_emb[i] = img_emb.view(bs, -1)

        frames_emb = self.pos(frames_emb) 
        out = self.transformer(src=frames_emb, src_key_padding_mask=mask)
        out = self.fc(self.dropout(torch.max(out, 0)[0]))
 
        return out 