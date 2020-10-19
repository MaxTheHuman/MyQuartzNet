import torch
import torchaudio
import pandas as pd
import multiprocessing as mp
import configparser
from torch import nn
from torch.utils.data import Dataset, DataLoader

from char_int_dicts import char2int, int2char
from model import ASR


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")
# print(device)

config = configparser.ConfigParser()
config.read('config.ini')

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    f_min=0,
    f_max=8000,
    n_mels=64).to(device)


def int_to_letters(text_in_ints):
    string = []
    for i in text_in_ints:
        string.append(int2char[i])
    return ''.join(string)

def Decoder(output, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_letters(decode))
    return decodes

log_softmax = nn.LogSoftmax(dim=2)

def apply(model, spectrogram):
    model.eval()
    with torch.no_grad():
        output = model(spectrogram)  # (batch, time, n_class)
        output = log_softmax(output)
        output = output.transpose(0, 1) # (time, batch, n_class)
        decoded_prediction = Decoder(output.transpose(0, 1))
        # print(decoded_preds)
    return decoded_prediction


model = ASR()
state_dict = torch.load(config.get('paths', 'path_to_weights_dict'))
model.load_state_dict(state_dict)
model = model.to(device)

waveform, sample_rate = torchaudio.load(config.get('paths', 'path_to_audio'))
spectrogram = mel_transform(waveform)
spectrogram = torch.log(spectrogram + 1e-9)

prediction = apply(model, spectrogram)
print(prediction[0])