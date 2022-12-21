"""Audio util file

TODO: find out where I got this from...
"""

import torch
import matplotlib.pyplot as plt
import random
import torch
import torchaudio
from torchaudio import transforms

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


def save_audio(
            self,
            waveform_real, 
            waveform_fake, 
            sample_rate,
            title="Waveform",
            xlim=None, 
            ylim=None
        ):
    waveform_real = waveform_real.numpy()
    waveform_fake = waveform_fake.numpy()

    num_channels, num_frames = waveform_real.shape

    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(nrows=num_channels, ncols=2)

    for ind in [0, 1]:
        label = {0: 'Real', 1: 'Fake'}[ind]
        waveform = {0: waveform_real, 1: waveform_fake}[ind]
        axes[1, ind].set_xlabel(label)
        for c in range(num_channels):
            axes[c, ind].plot(time_axis, waveform[c], linewidth=1)
            axes[c, ind].grid(True)

            if num_channels > 1:
                axes[c, ind].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c, ind].set_xlim(xlim)
            if ylim:
                axes[c, ind].set_ylim(ylim)

    figure.suptitle(title)
    fname = self.training_artifcat_path / f'{self.iter_count}.png'
    plt.savefig(fname)
