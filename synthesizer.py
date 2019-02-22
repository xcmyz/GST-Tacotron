from Network import *
from utils import audio
from text import text_to_sequence
import hparams

import numpy as np
import os


def synthesizer(model, text, ref_wav, device):
    seq = text_to_sequence(text, [hparams.cleaners])
    seq = np.stack([seq])
    if torch.cuda.is_available():
        seq = torch.from_numpy(seq).type(torch.cuda.LongTensor).to(device)
    else:
        seq = torch.from_numpy(seq).type(torch.LongTensor).to(device)
    # print(seq)

    # Provide [GO] Frame
    mel_input = np.zeros(
        [np.shape(seq)[0], 1, hparams.num_mels], dtype=np.float32)
    mel_input = torch.Tensor(mel_input).to(device)
    # print(np.shape(mel_input))

    ref_wav = np.stack([ref_wav[1:, :]])
    ref_wav = torch.Tensor(ref_wav).to(device)

    model.eval()
    with torch.no_grad():
        _, linear_output, _ = model(seq, mel_input, ref_wav)

    # print(np.shape(linear_output))
    wav = audio.inv_spectrogram(linear_output[0].cpu().numpy())
    wav = wav[:audio.find_endpoint(wav)]

    return wav


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron()).to(device)
    print("Model Have Been Defined")

    # Load checkpoint
    checkpoint = torch.load(os.path.join(
        hparams.checkpoint_path, 'checkpoint_20.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    print("Load Done")

    text = "I am very glad to meet you now."
    ref_name = os.path.join("dataset", "ljspeech-mel-00001.npy")
    ref_wav = np.load(ref_name)

    wav = synthesizer(model, text, ref_wav, device)
    audio.save_wav(wav, "test.wav")
