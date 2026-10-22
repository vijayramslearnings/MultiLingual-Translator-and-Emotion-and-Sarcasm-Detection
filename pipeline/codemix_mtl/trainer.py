"""
Multi-Task Training & Fine-Tuning Pipeline for Indian Code-Mixed Data.
Trains MuRIL Multi-Task Incongruity Model using:
- Class-Weighted Cross-Entropy for Sarcasm & Emotion
- Supervised Contrastive Incongruity Loss
- Benchmark Datasets: HASOC, SemEval Sentimix, DravidianLangTech
"""

import os
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import MuRILMultiTaskIncongruityModel, MTLModelConfig
from .loss import MultiTaskIncongruityLoss
from .datasets import (
    BaseCodeMixedDataset,
    CodeMixedDataCollator,
    compute_class_weights,
    create_synthetic_benchmarks
)


class MTLTrainer:
    """
    Production-ready fine-tuning trainer for MuRIL Multi-Task Architecture.
    Mitigates class imbalance via class-weighted loss and enforces semantic
    divergence for sarcastic expressions through metric contrastive learning.
    """
    def __init__(
        self,
        model: MuRILMultiTaskIncongruityModel,
        train_dataset: BaseCodeMixedDataset,
        val_dataset: Optional[BaseCodeMixedDataset] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        device: Optional[str] = None,
        lambda_emotion: float = 1.0,
        lambda_sarcasm: float = 1.5,
        lambda_contrastive: float = 0.8
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.batch_size = batch_size

        # DataLoaders with custom collation
        collator = CodeMixedDataCollator()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator
        ) if val_dataset else None

        # Compute dynamic class weights to mitigate sarcasm class imbalance
        sarcasm_labels = [s.get("sarcasm_label", 0) for s in train_dataset.samples]
        sarc_weights = compute_class_weights(sarcasm_labels, num_classes=2).to(self.device)

        emotion_labels = [s.get("emotion_label", 2) for s in train_dataset.samples]
        emo_weights = compute_class_weights(emotion_labels, num_classes=6).to(self.device)

        # Multi-task loss criterion
        self.criterion = MultiTaskIncongruityLoss(
            emotion_weights=emo_weights,
            sarcasm_weights=sarc_weights,
            contrastive_margin=model.cfg.contrastive_margin,
            lambda_emotion=lambda_emotion,
            lambda_sarcasm=lambda_sarcasm,
            lambda_contrastive=lambda_contrastive
        )

        # Optimizer with differential weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    def train_epoch(self, epoch_idx: int) -> Dict[str, float]:
        """Runs a single training epoch across code-mixed batches."""
        self.model.train()
        total_loss_accum = 0.0
        emo_loss_accum = 0.0
        sarc_loss_accum = 0.0
        cont_loss_accum = 0.0
        total_steps = len(self.train_loader)

        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            aux_features = batch["aux_features"].to(self.device)
            emotion_targets = batch["emotion_labels"].to(self.device)
            sarcasm_targets = batch["sarcasm_labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aux_features=aux_features
            )

            losses = self.criterion(
                model_outputs=outputs,
                emotion_targets=emotion_targets,
                sarcasm_targets=sarcasm_targets
            )

            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss_accum += loss.item()
            emo_loss_accum += losses["loss_emotion"].item()
            sarc_loss_accum += losses["loss_sarcasm"].item()
            cont_loss_accum += losses["loss_contrastive"].item()

        elapsed = time.time() - start_time
        metrics = {
            "epoch": epoch_idx,
            "total_loss": total_loss_accum / max(1, total_steps),
            "emotion_loss": emo_loss_accum / max(1, total_steps),
            "sarcasm_loss": sarc_loss_accum / max(1, total_steps),
            "contrastive_loss": cont_loss_accum / max(1, total_steps),
            "time_sec": elapsed
        }
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluates model performance and incongruity distance on validation set."""
        if not self.val_loader:
            return {}

        self.model.eval()
        emo_correct = 0
        sarc_correct = 0
        total_samples = 0
        literal_dists = []
        sarcastic_dists = []
        val_total_loss = 0.0
        val_emo_loss = 0.0
        val_sarc_loss = 0.0
        val_cont_loss = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                aux_features = batch["aux_features"].to(self.device)
                emotion_targets = batch["emotion_labels"].to(self.device)
                sarcasm_targets = batch["sarcasm_labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aux_features=aux_features
                )

                losses = self.criterion(
                    model_outputs=outputs,
                    emotion_targets=emotion_targets,
                    sarcasm_targets=sarcasm_targets
                )
                val_total_loss += losses["total_loss"].item()
                val_emo_loss += losses["loss_emotion"].item()
                val_sarc_loss += losses["loss_sarcasm"].item()
                val_cont_loss += losses["loss_contrastive"].item()

                emo_preds = torch.argmax(outputs["emotion_probs"], dim=-1)
                sarc_preds = torch.argmax(outputs["sarcasm_probs"], dim=-1)

                emo_correct += (emo_preds == emotion_targets).sum().item()
                sarc_correct += (sarc_preds == sarcasm_targets).sum().item()
                total_samples += input_ids.shape[0]

                # Collect incongruity distances per class
                dists = outputs["incongruity_distance"].squeeze(-1).cpu().numpy()
                sarc_labels = sarcasm_targets.cpu().numpy()
                for d, y in zip(dists, sarc_labels):
                    if y == 1:
                        sarcastic_dists.append(d)
                    else:
                        literal_dists.append(d)

        return {
            "val_loss": val_total_loss / max(1, total_batches),
            "val_emotion_loss": val_emo_loss / max(1, total_batches),
            "val_sarcasm_loss": val_sarc_loss / max(1, total_batches),
            "val_contrastive_loss": val_cont_loss / max(1, total_batches),
            "emotion_accuracy": emo_correct / max(1, total_samples),
            "sarcasm_accuracy": sarc_correct / max(1, total_samples),
            "mean_literal_distance": float(np.mean(literal_dists)) if literal_dists else 0.0,
            "mean_sarcastic_distance": float(np.mean(sarcastic_dists)) if sarcastic_dists else 0.0,
            "distance_separation_delta": float(
                (np.mean(sarcastic_dists) - np.mean(literal_dists))
            ) if (sarcastic_dists and literal_dists) else 0.0
        }

    def save_checkpoint(self, path: str):
        """Saves model weights and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": vars(self.model.cfg)
        }, path)
        print(f"[Trainer] Checkpoint saved successfully to: {path}")

    def load_checkpoint(self, path: str):
        """Loads fine-tuned model checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[Trainer] Checkpoint loaded successfully from: {path}")
