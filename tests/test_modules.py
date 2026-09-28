"""
Unit tests for each module independently.
"""

import os
import pytest
import torch
from pipeline.utils.audio_utils import generate_synthetic_audio, extract_acoustic_features
from pipeline.modules.module1_asr import SpeechRecognizer
from pipeline.modules.module2_lid import LanguageIdentifier
from pipeline.modules.module3_transliteration import TextNormalizer
from pipeline.modules.module4_mt import MachineTranslator
from pipeline.modules.module5_emotion import TextEmotionClassifier, AcousticEmotionClassifier, EmotionFusionLayer
from pipeline.modules.module6_sarcasm import SarcasmDetector
from pipeline.modules.module7_xai import ExplainableAI


@pytest.fixture
def mock_config():
    return {
        "device": "cpu",
        "pipeline": {"enable_transliteration": True},
        "asr": {"backend": "mock", "model_name": "openai/whisper-tiny", "sample_rate": 16000},
        "lid": {"backend": "heuristic"},
        "transliteration": {"backend": "indic_transliteration"},
        "mt": {"backend": "mock"},
        "emotion_text": {"labels": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]},
        "emotion_acoustic": {"labels": ["neutral", "happy", "sad", "angry"]},
        "fusion": {"text_weight": 0.55, "acoustic_weight": 0.45},
        "sarcasm": {"incongruity_threshold": 0.40},
        "xai": {"top_k_tokens": 5}
    }


@pytest.fixture
def synthetic_audio_path(tmp_path):
    path = str(tmp_path / "test_sample.wav")
    generate_synthetic_audio(path, duration_sec=2.0, tone_type="neutral")
    return path


def test_module1_asr(mock_config, synthetic_audio_path):
    asr = SpeechRecognizer(mock_config)
    res = asr.transcribe(synthetic_audio_path)
    assert "transcript" in res
    assert "word_timestamps" in res
    assert "confidence" in res
    assert isinstance(res["transcript"], str)
    assert isinstance(res["word_timestamps"], list)


def test_module2_lid(mock_config):
    lid = LanguageIdentifier(mock_config)
    res = lid.identify("Aapka kaam is really bahut badhiya hai")
    assert "language_segments" in res
    assert "detected_languages" in res
    assert "hi" in res["detected_languages"] or "en" in res["detected_languages"]
    assert res["is_code_mixed"] is True


def test_module3_transliteration(mock_config):
    normalizer = TextNormalizer(mock_config)
    segments = [{"text": "Aapka", "lang": "hi"}, {"text": "kaam", "lang": "hi"}]
    res = normalizer.normalize("Aapka kaam", segments)
    assert isinstance(res, str)
    assert len(res) > 0


def test_module4_mt(mock_config):
    translator = MachineTranslator(mock_config)
    res = translator.translate("Aapka kaam is really bahut badhiya hai", target_lang="en")
    assert isinstance(res, str)
    assert "work" in res.lower() or "excellent" in res.lower() or "badhiya" in res.lower()


def test_module5_emotion(mock_config, synthetic_audio_path):
    text_clf = TextEmotionClassifier(mock_config)
    acous_clf = AcousticEmotionClassifier(mock_config)
    fusion_layer = EmotionFusionLayer(mock_config)

    t_res = text_clf.predict("Aapka kaam is really bahut badhiya hai")
    assert "label" in t_res
    assert "confidence" in t_res

    waveform = torch.zeros(16000 * 2)
    a_res = acous_clf.predict(synthetic_audio_path, waveform)
    assert "label" in a_res
    assert "confidence" in a_res

    f_res = fusion_layer.fuse(t_res, a_res)
    assert "label" in f_res
    assert "modality_agreement" in f_res
    assert isinstance(f_res["modality_agreement"], bool)


def test_module6_sarcasm(mock_config):
    detector = SarcasmDetector(mock_config)
    t_res = {"label": "joy", "confidence": 0.85}
    a_res = {"label": "anger", "confidence": 0.80}
    f_res = {"label": "joy", "confidence": 0.70}

    res = detector.detect(t_res, a_res, f_res)
    assert "present" in res
    assert "incongruity_score" in res
    assert "reason" in res
    assert res["present"] is True  # High mismatch between joy text & anger acoustic


def test_module7_xai(mock_config):
    xai = ExplainableAI(mock_config)
    res = xai.generate_explanations(
        transcript="Aapka kaam is really bahut badhiya hai",
        emotion_label="joy",
        acoustic_emotion_label="joy",
        acoustic_features={"pitch_variance": 0.05, "max_energy_spike": 0.6, "duration_sec": 2.0}
    )
    assert "top_attributed_tokens" in res
    assert "acoustic_cues" in res
    assert len(res["top_attributed_tokens"]) > 0
    assert len(res["acoustic_cues"]) > 0
