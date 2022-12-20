"""Audio util file

TODO: find out where I got this from...
"""


import random
import torch
import torchaudio
from torchaudio import transforms
# from IPython.display import Audio

# todo: randomize this for each __get_item__ call
FRAME_OFFSET = 16000


class AudioUtil():
    @staticmethod
    def open(audio_file, num_frames):
        sig, sr = torchaudio.load(
            audio_file, 
            frame_offset=FRAME_OFFSET, 
            num_frames=num_frames,
            format="mp3")
        return (sig, sr)

    @staticmethod
    def rechannel(aud):
        sig, sr = aud
        if (sig.shape[0] == 2):
            return aud
        else:
            resig = torch.cat([sig, sig])
        return ((resig, sr))
      
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            return aud
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig)
        return ((resig, newsr))

    @staticmethod
    def random_portion(aud, size):
        sig, sr = aud
        i = random.randint(0, sig.shape[1] - size)
        return (sig[:, i:i+size], sr)


# import torch
# import matplotlib.pyplot as plt
# from IPython.display import Audio, display


# def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
#     waveform = waveform.numpy()

#     num_channels, num_frames = waveform.shape
#     time_axis = torch.arange(0, num_frames) / sample_rate

#     figure, axes = plt.subplots(num_channels, 1)
#     if num_channels == 1:
#         axes = [axes]
#     for c in range(num_channels):
#         axes[c].plot(time_axis, waveform[c], linewidth=1)
#         axes[c].grid(True)
#         if num_channels > 1:
#             axes[c].set_ylabel(f'Channel {c+1}')
#         if xlim:
#             axes[c].set_xlim(xlim)
#         if ylim:
#             axes[c].set_ylim(ylim)
#     figure.suptitle(title)
#     plt.show(block=False)

# def play_audio(waveform, sample_rate):
#     waveform = waveform.numpy()
#     num_channels, num_frames = waveform.shape
#     if num_channels == 1:
#         display(Audio(waveform[0], rate=sample_rate))
#     elif num_channels == 2:
#         display(Audio((waveform[0], waveform[1]), rate=sample_rate))
#     else:
#         raise ValueError("Waveform with more than 2 channels are not supported.")