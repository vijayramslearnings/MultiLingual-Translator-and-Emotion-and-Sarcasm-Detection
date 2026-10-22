"""
Integration tests for the master MultilingualVoicePipeline class.
Verifies end-to-end execution and JSON schema compliance.
"""

import os
import pytest
from pipeline import MultilingualVoicePipeline
from pipeline.utils.audio_utils import generate_synthetic_audio


@pytest.fixture
def sample_audio(tmp_path):
    audio_path = str(tmp_path / "test_utterance.wav")
    generate_synthetic_audio(audio_path, duration_sec=3.0, tone_type="sarcastic")
    return audio_path


def test_pipeline_end_to_end_json_schema(sample_audio):
    config_path = "config/default_config.yaml"
    pipeline = MultilingualVoicePipeline(config_path=config_path)

    result = pipeline.process(sample_audio, target_lang="en")

    # 1. Root keys check
    expected_root_keys = [
        "transcript", "language_segments", "translated_text", 
        "target_language", "emotion", "sarcasm", "explanation"
    ]
    for key in expected_root_keys:
        assert key in result, f"Missing root key in result JSON: '{key}'"

    # 2. Check Types
    assert isinstance(result["transcript"], str)
    assert isinstance(result["language_segments"], list)
    assert isinstance(result["translated_text"], str)
    assert isinstance(result["target_language"], str)
    assert result["target_language"] == "en"

    # 3. Language Segments Structure
    if len(result["language_segments"]) > 0:
        seg = result["language_segments"][0]
        assert "text" in seg
        assert "lang" in seg

    # 4. Emotion Object Structure
    emotion = result["emotion"]
    assert "label" in emotion
    assert "confidence" in emotion
    assert "text_modality" in emotion
    assert "acoustic_modality" in emotion
    assert "modality_agreement" in emotion
    assert isinstance(emotion["confidence"], float)
    assert isinstance(emotion["modality_agreement"], bool)
    assert "label" in emotion["text_modality"]
    assert "confidence" in emotion["text_modality"]
    assert "label" in emotion["acoustic_modality"]
    assert "confidence" in emotion["acoustic_modality"]

    # 5. Sarcasm Object Structure
    sarcasm = result["sarcasm"]
    assert "present" in sarcasm
    assert "confidence" in sarcasm
    assert "incongruity_score" in sarcasm
    assert "reason" in sarcasm
    assert isinstance(sarcasm["present"], bool)
    assert isinstance(sarcasm["confidence"], float)
    assert isinstance(sarcasm["incongruity_score"], float)
    assert isinstance(sarcasm["reason"], str)

    # 6. Explanation Object Structure
    explanation = result["explanation"]
    assert "top_attributed_tokens" in explanation
    assert "acoustic_cues" in explanation
    assert isinstance(explanation["top_attributed_tokens"], list)
    assert isinstance(explanation["acoustic_cues"], list)

    if len(explanation["top_attributed_tokens"]) > 0:
        attr = explanation["top_attributed_tokens"][0]
        assert "token" in attr
        assert "score" in attr
        assert isinstance(attr["score"], float)


def test_pipeline_tanglish_source_lang(sample_audio):
    config_path = "config/default_config.yaml"
    pipeline = MultilingualVoicePipeline(config_path=config_path)

    # Tanglish input with explicit source_lang="ta-en"
    tanglish_text = "hi nan iinaiku party ku poren so nalaiku nan varapa nee yen homework mattum pandriya please??"
    result = pipeline.process(
        sample_audio, 
        target_lang="en", 
        source_lang="ta-en",
        override_transcript=tanglish_text
    )

    assert result["source_language"] == "ta-en"
    assert "homework" in result["translated_text"].lower()
    assert result["detected_dialect"] == "Tanglish (Tamil + English)"
    # Language segments must not have Hindi 'hi'
    for seg in result["language_segments"]:
        assert seg["lang"] in ["ta", "en"], f"Unexpected language tag {seg['lang']} for Tanglish token {seg['text']}"
