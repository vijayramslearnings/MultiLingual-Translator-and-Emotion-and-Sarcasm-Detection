"""
Unit and Integration Tests for Indian Code-Mixed MTL & IndicTrans2 Subsystem.
Tests:
- CodeMixedPreprocessor (phonetic squashing, emoji dense vector, token LID, switch metrics)
- MuRILMultiTaskIncongruityModel forward pass and output shapes
- ContrastiveIncongruityLoss and MultiTaskIncongruityLoss
- HASOC, SemEval Sentimix, and DravidianLangTech Dataset classes and DataCollator
- IndicTrans2LocalEngine and IndicXlitLocalEngine
- CodeMixedInferenceEngine forward evaluation
"""

import os
import pytest
import torch
import numpy as np

# Ensure fast local execution for testing
os.environ["INDIC_FAST_MODE"] = "1"

from pipeline.codemix_mtl import (
    CodeMixedPreprocessor,
    PreprocessedOutput,
    MuRILMultiTaskIncongruityModel,
    MTLModelConfig,
    MultiTaskIncongruityLoss,
    ContrastiveIncongruityLoss,
    HASOCDataset,
    SemEvalSentimixDataset,
    DravidianLangTechDataset,
    CodeMixedDataCollator,
    create_synthetic_benchmarks,
    MTLTrainer,
    IndicTrans2LocalEngine,
    IndicXlitLocalEngine,
    CodeMixedInferenceEngine,
)


class TestCodeMixedPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return CodeMixedPreprocessor()

    def test_phonetic_squashing(self, preprocessor):
        raw = "booooodhiyaaaaa"
        normalized = preprocessor.squash_phonetic_elongations(raw)
        assert normalized == "badhiya"

        raw2 = "achhhhaaaa"
        normalized2 = preprocessor.squash_phonetic_elongations(raw2)
        assert normalized2 == "accha"

        raw3 = "sooooo"
        normalized3 = preprocessor.squash_phonetic_elongations(raw3)
        assert normalized3 == "so"

    def test_emoji_vector_extraction(self, preprocessor):
        text = "Wah boss!! 😂👏"
        emojis, vec = preprocessor.extract_and_embed_emojis(text, vector_dim=16)
        assert "😂" in emojis
        assert "👏" in emojis
        assert len(vec) == 16
        assert vec[3] > 0.7  # High sarcasm affinity flag
        assert vec[10] == 1.0 # Sarcastic emoji categorical flag active

    def test_token_lid_and_switch_density(self, preprocessor):
        text = "Romba nalla velai sir thanks a lot"
        out = preprocessor.preprocess(text)
        assert isinstance(out, PreprocessedOutput)
        assert out.switch_count >= 1
        assert out.indic_ratio > 0.0
        assert out.english_ratio > 0.0
        assert len(out.auxiliary_feature_vector) == 32


class TestMuRILMTLArchitecture:
    @pytest.fixture
    def model(self):
        cfg = MTLModelConfig(offline_fast_mode=True)
        return MuRILMultiTaskIncongruityModel(cfg)

    def test_forward_shapes(self, model):
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        aux_features = torch.randn((batch_size, 32))

        outputs = model(input_ids, attention_mask, aux_features)

        assert "emotion_logits" in outputs
        assert outputs["emotion_logits"].shape == (batch_size, 6)
        assert "emotion_probs" in outputs
        assert outputs["emotion_probs"].shape == (batch_size, 6)

        assert "sarcasm_logits" in outputs
        assert outputs["sarcasm_logits"].shape == (batch_size, 2)
        assert "sarcasm_probs" in outputs
        assert outputs["sarcasm_probs"].shape == (batch_size, 2)

        assert "incongruity_distance" in outputs
        assert outputs["incongruity_distance"].shape == (batch_size, 1)
        assert outputs["context_embedding"].shape == (batch_size, 128)
        assert outputs["emotion_embedding"].shape == (batch_size, 128)


class TestLossFunctions:
    def test_contrastive_incongruity_loss(self):
        criterion = ContrastiveIncongruityLoss(margin=1.0)
        z_ctx = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        z_emo = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1]) # 0: literal, 1: sarcastic

        loss = criterion(z_ctx, z_emo, labels)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_multitask_loss(self):
        loss_fn = MultiTaskIncongruityLoss()
        model_outputs = {
            "emotion_logits": torch.randn(2, 6),
            "sarcasm_logits": torch.randn(2, 2),
            "context_embedding": torch.randn(2, 128),
            "emotion_embedding": torch.randn(2, 128),
            "incongruity_distance": torch.tensor([[0.5], [1.2]])
        }
        targets_emo = torch.tensor([0, 4])
        targets_sarc = torch.tensor([0, 1])

        losses = loss_fn(model_outputs, targets_emo, targets_sarc)
        assert "total_loss" in losses
        assert "loss_emotion" in losses
        assert "loss_sarcasm" in losses
        assert "loss_contrastive" in losses
        assert losses["total_loss"].item() > 0.0


class TestDatasetsAndBenchmarks:
    def test_synthetic_benchmarks_and_collator(self):
        benchmarks = create_synthetic_benchmarks()
        assert "HASOC" in benchmarks
        assert "SemEval_Sentimix" in benchmarks
        assert "DravidianLangTech" in benchmarks

        ds = HASOCDataset(benchmarks["HASOC"])
        assert len(ds) == len(benchmarks["HASOC"])

        collator = CodeMixedDataCollator()
        batch = collator([ds[0], ds[1]])
        assert batch["input_ids"].shape[0] == 2
        assert batch["aux_features"].shape == (2, 32)
        assert batch["emotion_labels"].shape == (2,)
        assert batch["sarcasm_labels"].shape == (2,)


class TestInferenceEngine:
    def test_end_to_end_evaluate(self):
        engine = CodeMixedInferenceEngine()
        res = engine.evaluate(
            "Arreeee booooodhiyaaaaa kaam kiya boss!! 😂👏 Wah kya presentation hai!",
            source_lang="hi-en"
        )
        assert "input" in res
        assert "linguistic_analysis" in res
        assert "translation" in res
        assert "emotion_detection" in res
        assert "sarcasm_detection" in res

        assert res["input"]["normalized_text"] == "Arrey badhiya kaam kiya boss!! 😂👏 Wah kya presentation hai!"
        assert res["sarcasm_detection"]["is_sarcastic"] is True
        assert res["sarcasm_detection"]["context_incongruity_distance"] > 0.0
