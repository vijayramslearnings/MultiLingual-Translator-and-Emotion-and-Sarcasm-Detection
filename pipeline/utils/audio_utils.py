"""
Audio utilities for loading, resampling, synthetic audio generation, 
and acoustic feature extraction (spectrogram, energy, pitch).
Uses standard wave/scipy for rock-solid audio saving/loading across Python versions.
"""

import os
import wave
import numpy as np
import torch
import torchaudio


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and resample it to target_sr.
    Returns (waveform, sample_rate) where waveform is a 1D or 2D torch.Tensor.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # Try standard wave module first for WAV files
    try:
        with wave.open(audio_path, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            
            sampwidth = wf.getsampwidth()
            if sampwidth == 2:
                np_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                np_data = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                np_data = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if n_channels > 1:
                np_data = np_data.reshape(-1, n_channels).mean(axis=1)

            waveform = torch.tensor(np_data, dtype=torch.float32)
    except Exception:
        # Fallback to torchaudio.load
        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform.squeeze(), sr


def extract_acoustic_features(waveform: torch.Tensor, sample_rate: int = 16000) -> dict:
    """
    Extract frame-level acoustic features: Mel-spectrogram, RMS energy, and Pitch variance.
    Used by acoustic emotion detection and XAI modules.
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    np_wave = waveform.numpy() if isinstance(waveform, torch.Tensor) else np.array(waveform)

    # 1. Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80
    )
    mel_spec = mel_transform(torch.tensor(np_wave, dtype=torch.float32))
    log_mel_spec = torch.log(mel_spec + 1e-6)

    # 2. RMS Energy across time frames
    frame_length = 1024
    hop_length = 512
    num_frames = max(1, (len(np_wave) - frame_length) // hop_length + 1)
    energy = []
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(np_wave))
        chunk = np_wave[start:end]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
        energy.append(rms)
    energy = np.array(energy)

    # 3. Spectral Centroid proxy
    spec_centroid = np.mean(log_mel_spec.numpy(), axis=0) if log_mel_spec.ndim > 1 else np.zeros(num_frames)

    mean_energy = float(np.mean(energy)) if len(energy) > 0 else 0.0
    std_energy = float(np.std(energy)) if len(energy) > 0 else 0.0
    energy_spikes = float(np.max(energy)) if len(energy) > 0 else 0.0
    pitch_variance = float(np.var(spec_centroid)) if len(spec_centroid) > 0 else 0.0

    return {
        "log_mel_spectrogram": log_mel_spec,
        "energy_contour": energy,
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "max_energy_spike": energy_spikes,
        "pitch_variance": pitch_variance,
        "duration_sec": len(np_wave) / sample_rate
    }


def generate_synthetic_audio(output_path: str, duration_sec: float = 3.0, sample_rate: int = 16000, tone_type: str = "neutral") -> str:
    """
    Generate speech test audio wav file.
    Prefers realistic spoken utterance generation (via macOS TTS if available),
    falling back to acoustic sine/harmonic synthesis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Try realistic speech synthesis using system TTS (macOS say + afconvert)
    try:
        import shutil
        import subprocess
        if shutil.which("say") and shutil.which("afconvert"):
            phrases = {
                "sarcastic": ("Lekha", "Aapka kaam is really bahut badhiya hai boss"),
                "happy": ("Vani", "Romba nalla velai sir thanks a lot"),
                "angry": ("Lekha", "Ye project report bilkul bekar hai"),
                "neutral": ("Lekha", "Aapka kaam bahut accha hai thank you")
            }
            voice, phrase = phrases.get(tone_type, ("Lekha", "Aapka kaam is really bahut badhiya hai"))
            temp_aiff = output_path.replace(".wav", "_temp.aiff")
            res_say = subprocess.run(["say", "-v", voice, phrase, "-o", temp_aiff], capture_output=True)
            if res_say.returncode == 0 and os.path.exists(temp_aiff):
                res_conv = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", temp_aiff, output_path], capture_output=True)
                if os.path.exists(temp_aiff):
                    os.remove(temp_aiff)
                if res_conv.returncode == 0 and os.path.exists(output_path):
                    return output_path
    except Exception:
        pass

    # 2. Harmonic waveform fallback
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    if tone_type == "happy":
        f0 = 220 + 50 * np.sin(2 * np.pi * 3 * t)
        signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t)
    elif tone_type == "angry":
        f0 = 150 + 80 * np.sin(2 * np.pi * 5 * t)
        signal = 0.8 * np.sin(2 * np.pi * f0 * t) + 0.5 * np.random.normal(0, 0.1, len(t))
    elif tone_type == "sarcastic":
        f0 = 180 + 120 * np.sin(2 * np.pi * 1.5 * t) ** 3
        signal = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(6 * np.pi * f0 * t)
    else:  # neutral
        f0 = 160 + 10 * np.sin(2 * np.pi * 1 * t)
        signal = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 320 * t)

    envelope = np.ones_like(t)
    fade_len = int(0.05 * sample_rate)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    final_signal = (signal * envelope * 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(final_signal.tobytes())

    return output_path
