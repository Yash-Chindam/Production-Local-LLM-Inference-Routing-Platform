"""Local LLM inference gateway and routing policy package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-llm-router")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0.0.0"
