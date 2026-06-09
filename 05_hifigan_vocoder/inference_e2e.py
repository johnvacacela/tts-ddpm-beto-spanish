"""
Inferencia HiFi-GAN mel→audio.

Este script se utiliza únicamente para vocodificación con un checkpoint
HiFi-GAN preentrenado. En la tesis no se entrenó ni ajustó HiFi-GAN;
el vocoder se mantuvo fijo para Baseline FastPitch y DDPM+BETO.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.io.wavfile import write

from env import AttrDict
from models import Generator

MAX_WAV_VALUE = 32768.0

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    """Convierte los Mels de entrada en archivos WAV."""
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = sorted(
        filename for filename in os.listdir(a.input_mels_dir)
        if filename.lower().endswith('.npy')
    )
    if not filelist:
        raise FileNotFoundError(
            f"No se encontraron archivos .npy en {a.input_mels_dir}")

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for filename in filelist:
            x = np.load(
                os.path.join(a.input_mels_dir, filename),
                allow_pickle=False,
            )
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(
                a.output_dir,
                os.path.splitext(filename)[0] + '_generated_e2e.wav',
            )
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser(
        description='Vocodifica Mels con un checkpoint HiFi-GAN preentrenado')
    parser.add_argument(
        '--input_mels_dir', required=True,
        help='Directorio con Mel-espectrogramas .npy')
    parser.add_argument(
        '--output_dir', required=True,
        help='Directorio para los audios .wav generados')
    parser.add_argument(
        '--checkpoint_file', required=True,
        help='Checkpoint universal preentrenado de HiFi-GAN')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"No se encontró config.json junto al checkpoint: {config_file}")
    with open(config_file, encoding='utf-8') as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
