"""Descriptors only. Runtime discovery/assertion belongs to Sprint 5."""

from .base import DurableModel
from .models import PositiveInt
from .refs import StackId


class CapabilitySpec(DurableModel):
    dtypes: tuple[str, ...]
    quantizations: tuple[str, ...]
    max_context_tokens: PositiveInt
    lora: bool
    training_backends: tuple[str, ...]
    generation_backends: tuple[str, ...]
    max_tensor_parallel_size: PositiveInt
    importance_sampling_modes: tuple[str, ...]
    colocated_generation: bool


class CapabilityReport(DurableModel):
    backend_id: str
    stack_id: StackId
    capabilities: CapabilitySpec
    compatible: bool
    reasons: tuple[str, ...] = ()
