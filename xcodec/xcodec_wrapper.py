import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import librosa
from omegaconf import OmegaConf
import torch
import torch.nn as nn
 
from models.soundstream_semantic import SoundStream


def load_codec_model(config_path, model_path):
    config = OmegaConf.load(config_path)
    model = eval(config.generator.name)(**config.generator.config)
    parameter_dict = torch.load(model_path)
    model.load_state_dict(parameter_dict)
    return model


class XCodecFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class UnitTest:
    def __init__(self):
        self.model_path = "/fsx/workspace/martin/code/xcodec/speech_ckpt/general/xcodec_hubert_general_audio_v2.pth"
        self.config_path = os.path.join(os.path.dirname(self.model_path), "config.yaml")
        self.tokenizer = load_codec_model(self.config_path, self.model_path)
        self.feature_extractor = XCodecFeatureExtractor(sampling_rate=16000)

    def test_encode_decode(self):
        audio_signal = torch.randn(1, 1, 320000)
        encoded = self.tokenizer.encode(audio_signal)
        encoded = encoded.audio_codes
        decoded = self.tokenizer.decode(encoded)
        assert audio_signal.shape == decoded.shape
    
    def test_feature_extractor(self):
        audio_signal, sr = librosa.load("/fsx/workspace/martin/code/xcodec/test_audio/music.wav", sr=16000)
        feature_extractor = XCodecFeatureExtractor(sampling_rate=16000)
        audio_signal = feature_extractor(audio_signal, 16000, "pt")
        assert audio_signal["input_values"].shape == (1, 1, 320000)


if __name__ == "__main__":
    unittest = UnitTest()
    unittest.test_encode_decode()
    unittest.test_feature_extractor()