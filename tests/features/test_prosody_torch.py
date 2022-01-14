import pytest
import torch

from datasets_turntaking.features.transforms import ProsodyTorch


@pytest.mark.features
@pytest.mark.transforms
@pytest.mark.parametrize("sample_rate", [16000, 8000])
def test_encoder(sample_rate):

    extractor = ProsodyTorch(sample_rate=sample_rate, frame_time=0.05, hop_time=0.01)
    waveform = torch.randn((1, sample_rate * 2))
    features = extractor(waveform)

    assert tuple(features.shape) == (1, 201, extractor.n_features)
