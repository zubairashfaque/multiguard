"""Knowledge distillation trainer for model compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer import MultimodalTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DistillationTrainer(MultimodalTrainer):
    """Trainer for knowledge distillation from a large teacher to a small student.

    Uses a combination of hard label loss and soft KL-divergence loss.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        **kwargs: dict,
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.temperature = temperature
        self.alpha = alpha
        logger.info(f"DistillationTrainer: temp={temperature}, alpha={alpha}")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined hard + soft distillation loss.

        Args:
            student_logits: Student predictions [B, C].
            teacher_logits: Teacher predictions [B, C].
            labels: Ground truth labels [B].

        Returns:
            Combined scalar loss.
        """
        hard_loss = F.cross_entropy(student_logits, labels)

        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        soft_loss = soft_loss * (self.temperature**2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
