"""Knowledge distillation trainer for model compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.training.trainer import MultimodalTrainer
from src.utils.device import log_vram_usage
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
        self.teacher_model.to(self.device)
        self.temperature = temperature
        self.alpha = alpha
        logger.info(f"DistillationTrainer: temp={temperature}, alpha={alpha}")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined hard + soft distillation loss."""
        hard_loss = F.cross_entropy(student_logits, labels)

        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        soft_loss = soft_loss * (self.temperature**2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one distillation training epoch.

        Overrides parent to use teacher model for soft targets.
        """
        self.model.train()
        self.teacher_model.eval()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    student_outputs = self.model(batch)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(batch)
                    loss = self.compute_distillation_loss(
                        student_outputs["logits"],
                        teacher_outputs["logits"],
                        batch["labels"],
                    )
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1
            else:
                student_outputs = self.model(batch)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch)
                loss = self.compute_distillation_loss(
                    student_outputs["logits"],
                    teacher_outputs["logits"],
                    batch["labels"],
                )
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            if self.global_step > 0 and self.global_step % self.logging_steps == 0:
                step_loss = total_loss / num_batches
                self.wandb_cb.log(
                    {"train/loss": step_loss, "train/lr": self.optimizer.param_groups[0]["lr"]},
                    self.global_step,
                )

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f} (distillation)")
        log_vram_usage(f"epoch_{epoch}")
        return {"train_loss": avg_loss}
