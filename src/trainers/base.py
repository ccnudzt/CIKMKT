from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def fit(self, train_loader, val_loader) -> dict:
        """Train the model and return summary metrics."""
