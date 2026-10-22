"""
Custom Loss Functions for Multi-Task Learning on Code-Mixed Data.
Includes:
1. ContrastiveIncongruityLoss: Mathematical metric distance separation between context & literal sentiment
2. ClassWeightedCrossEntropyLoss: Clean mitigation of severe class imbalance in sarcasm datasets
3. MultiTaskIncongruityLoss: Unified multi-task objective with dynamic loss weighting
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveIncongruityLoss(nn.Module):
    """
    Supervised Contrastive Loss resolving Sentiment Context Incongruity.
    Pulls together literal representations (y = 0 -> distance D -> 0)
    and pushes apart sarcastic representations (y = 1 -> distance D >= margin).
    """
    def __init__(self, margin: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        z_context: torch.Tensor,
        z_emotion: torch.Tensor,
        sarcasm_labels: torch.Tensor,
        incongruity_distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z_context: (batch, dim) normalized context embedding
            z_emotion: (batch, dim) normalized emotion embedding
            sarcasm_labels: (batch,) integer labels (0: literal, 1: sarcastic)
            incongruity_distance: optional precomputed (batch, 1) or (batch,) distance
        """
        if incongruity_distance is None:
            dist = torch.norm(z_context - z_emotion, p=2, dim=-1)
        else:
            dist = incongruity_distance.squeeze(-1)

        y = sarcasm_labels.float()

        # Literal loss (y=0): minimize distance
        loss_literal = (1.0 - y) * 0.5 * torch.pow(dist, 2)

        # Sarcastic loss (y=1): distance must exceed margin
        loss_sarcastic = y * 0.5 * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        loss = torch.mean(loss_literal + loss_sarcastic)
        return loss


class MultiTaskIncongruityLoss(nn.Module):
    """
    Unified Multi-Task Loss module combining:
    1. Emotion Cross-Entropy Loss (with optional class weights)
    2. Sarcasm Class-Weighted Cross-Entropy / Focal Loss (mitigating severe sarcasm class imbalance)
    3. Metric Contrastive Incongruity Loss
    """
    def __init__(
        self,
        emotion_weights: Optional[torch.Tensor] = None,
        sarcasm_weights: Optional[torch.Tensor] = None,
        contrastive_margin: float = 1.0,
        lambda_emotion: float = 1.0,
        lambda_sarcasm: float = 1.5,
        lambda_contrastive: float = 0.8
    ):
        super().__init__()
        # If sarcasm weights are not provided, default to 1:3 ratio reflecting
        # typical 75% literal : 25% sarcastic imbalance in social datasets
        if sarcasm_weights is None:
            sarcasm_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)

        self.emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights)
        self.sarcasm_criterion = nn.CrossEntropyLoss(weight=sarcasm_weights)
        self.contrastive_criterion = ContrastiveIncongruityLoss(margin=contrastive_margin)

        self.lambda_emotion = lambda_emotion
        self.lambda_sarcasm = lambda_sarcasm
        self.lambda_contrastive = lambda_contrastive

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        emotion_targets: Optional[torch.Tensor] = None,
        sarcasm_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total loss and individual task loss components.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=model_outputs["emotion_logits"].device)

        # 1. Emotion Task Loss
        if emotion_targets is not None:
            loss_emo = self.emotion_criterion(model_outputs["emotion_logits"], emotion_targets)
            losses["loss_emotion"] = loss_emo
            total_loss = total_loss + (self.lambda_emotion * loss_emo)
        else:
            losses["loss_emotion"] = torch.tensor(0.0)

        # 2. Sarcasm Task Loss (Class-Weighted)
        if sarcasm_targets is not None:
            loss_sarc = self.sarcasm_criterion(model_outputs["sarcasm_logits"], sarcasm_targets)
            losses["loss_sarcasm"] = loss_sarc
            total_loss = total_loss + (self.lambda_sarcasm * loss_sarc)

            # 3. Contrastive Incongruity Loss
            loss_cont = self.contrastive_criterion(
                z_context=model_outputs["context_embedding"],
                z_emotion=model_outputs["emotion_embedding"],
                sarcasm_labels=sarcasm_targets,
                incongruity_distance=model_outputs["incongruity_distance"]
            )
            losses["loss_contrastive"] = loss_cont
            total_loss = total_loss + (self.lambda_contrastive * loss_cont)
        else:
            losses["loss_sarcasm"] = torch.tensor(0.0)
            losses["loss_contrastive"] = torch.tensor(0.0)

        losses["total_loss"] = total_loss
        return losses
