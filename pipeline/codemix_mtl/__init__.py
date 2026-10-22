"""
Advanced Multilingual Translation, Emotion, and Sarcasm Detection System
Tailored for Indian Code-Mixed Languages (Hinglish, Tamlish, Bengalish, etc.)
Powered by MuRIL, IndicTrans2, IndicXlit, and Contrastive Incongruity MTL.
"""

from .preprocessor import CodeMixedPreprocessor, PreprocessedOutput
from .model import MuRILMultiTaskIncongruityModel, MTLModelConfig
from .loss import MultiTaskIncongruityLoss, ContrastiveIncongruityLoss
from .datasets import (
    HASOCDataset,
    SemEvalSentimixDataset,
    DravidianLangTechDataset,
    CodeMixedDataCollator,
    create_synthetic_benchmarks
)
from .trainer import MTLTrainer
from .indic_translation import IndicTrans2LocalEngine, IndicXlitLocalEngine
from .inference import CodeMixedInferenceEngine

__all__ = [
    "CodeMixedPreprocessor",
    "PreprocessedOutput",
    "MuRILMultiTaskIncongruityModel",
    "MTLModelConfig",
    "MultiTaskIncongruityLoss",
    "ContrastiveIncongruityLoss",
    "HASOCDataset",
    "SemEvalSentimixDataset",
    "DravidianLangTechDataset",
    "CodeMixedDataCollator",
    "create_synthetic_benchmarks",
    "MTLTrainer",
    "IndicTrans2LocalEngine",
    "IndicXlitLocalEngine",
    "CodeMixedInferenceEngine",
]
