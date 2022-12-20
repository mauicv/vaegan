from duct.utils.audio import AudioUtil

def test_audio_utils():
    (sig, sr) = AudioUtil.open('./datasets/fma_small/fma_small/000/000002.mp3', 2**13)
    assert sig.shape == (2, 2**13)
