"""Class registry pattern for dynamic model/component registration."""

from collections.abc import Callable
from typing import Any


class Registry:
    """A registry for mapping names to classes or factory functions.

    Usage:
        model_registry = Registry("models")

        @model_registry.register("clip")
        class CLIPBackbone:
            ...

        model_cls = model_registry.get("clip")
        model = model_cls(config)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[Any]] = {}

    def register(self, name: str) -> Callable:
        """Register a class or function under a name."""

        def decorator(cls: type[Any]) -> type[Any]:
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in {self._name} registry")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        """Get a registered class by name."""
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"'{name}' not found in {self._name} registry. Available: [{available}]")
        return self._registry[name]

    def list_registered(self) -> list[str]:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# Global registries
MODEL_REGISTRY = Registry("models")
BACKBONE_REGISTRY = Registry("backbones")
FUSION_REGISTRY = Registry("fusion")
HEAD_REGISTRY = Registry("heads")
TRAINER_REGISTRY = Registry("trainers")
BENCHMARK_REGISTRY = Registry("benchmarks")
