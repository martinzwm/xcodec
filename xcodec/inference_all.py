import argparse
import os
from pathlib import Path
import sys
from tqdm import tqdm
import multiprocessing
import torchaudio

import torch
import typing as tp
from omegaconf import OmegaConf
 
from models.soundstream_semantic import SoundStream
import torch.nn.functional as F

 
def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str], sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    
    path = str(Path(path).with_suffix('.wav'))
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

 
def process_audio(input_file, output_file, rescale, bw, config, soundstream, device):
    # Loading audio
    wav, sr = torchaudio.load(input_file)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)  # Convert to mono
    if sr != soundstream.sample_rate:
        wav = torchaudio.transforms.Resample(sr, soundstream.sample_rate)(wav)
    if config.audio_norm_scale < 1.0:
        wav = wav * config.audio_norm_scale
    
    wav = wav.unsqueeze(1).to(device)
    compressed = soundstream.encode(wav,   target_bw=bw)
    
    # Decode and save
    out = soundstream.decode(compressed)
    out = out.detach().cpu().squeeze(0)
 
    save_audio(out, output_file, 16000, rescale=rescale)


def process_folder(input_folder, output_folder, rescale, bw, config, model_path, device):
    torch.cuda.set_device(device)
    
    # Load model
    soundstream = build_codec_model(config)
    soundstream.load_state_dict(torch.load(model_path, map_location=device))
    soundstream.to(device)
    soundstream.eval()

    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'))]
    for file in tqdm(audio_files):
        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, file)
        process_audio(input_file, output_file, rescale, bw, config, soundstream, device)


def process_all_folders(original_folder, recon_folder, rescale, bw, config, model_path):
    num_gpus = torch.cuda.device_count()
    pool = multiprocessing.Pool(processes=num_gpus)
    tasks = [
        (os.path.join(original_folder, folder), os.path.join(recon_folder, folder), rescale, bw, config, model_path, f"cuda:{i % num_gpus}")
        for i, folder in enumerate(os.listdir(original_folder))
    ]
    pool.starmap(process_folder, tasks)

    pool.close()
    pool.join()


if __name__ == "__main__":
    # Parameters
    original_folder = "/fsx/workspace/martin/tokenizer_eval/original/"
    recon_folder = "/fsx/workspace/martin/tokenizer_eval/xcodec_8codes/"
    model_path = "/fsx/workspace/martin/code/xcodec/speech_ckpt/general/xcodec_hubert_general_audio_v2.pth"
    rescale = True
    bw = 4
    
    config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    config = OmegaConf.load(config_path)

    multiprocessing.set_start_method('spawn')
    process_all_folders(original_folder, recon_folder, rescale, bw, config, model_path)
