from __future__ import annotations


class Registry:
    """Minimal string-to-object registry."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, object] = {}

    def register(self, key: str):
        def decorator(obj: object) -> object:
            if key in self._items:
                raise KeyError(f"{self.name} registry already contains '{key}'.")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str):
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}") from exc

    def keys(self) -> list[str]:
        return sorted(self._items)
