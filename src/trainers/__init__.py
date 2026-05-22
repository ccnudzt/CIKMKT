"""Trainer registry and implementations."""

from src.core.registry import Registry

TRAINER_REGISTRY = Registry("trainer")

from src.trainers import sequence  # noqa: F401
