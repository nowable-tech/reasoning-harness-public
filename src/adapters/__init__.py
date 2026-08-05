from .base import BaseAdapter, CredentialMissingError, ModelResponse
from .deepseek import DeepSeekAdapter
from .zai import ZaiAdapter
from .moonshot import MoonshotAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemma import GemmaAdapter
from .minimax import MinimaxAdapter
from .google_adapter import GoogleAdapter
from .mistral_adapter import MistralAdapter
from .thinkingmachines import ThinkingMachinesAdapter

PROVIDER_MAP: dict[str, type[BaseAdapter]] = {
    "deepseek": DeepSeekAdapter,
    "zai": ZaiAdapter,
    "moonshot": MoonshotAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemma": GemmaAdapter,       # Gemma 4 via OpenRouter (was local/Ollama)
    "minimax": MinimaxAdapter,
    "google": GoogleAdapter,     # Gemini judge stub (Phase 2)
    "mistral": MistralAdapter,   # Mistral via OpenRouter
    "thinkingmachines": ThinkingMachinesAdapter,  # Inkling via OpenRouter (no direct API; third-party hosted)
}

__all__ = [
    "BaseAdapter",
    "CredentialMissingError",
    "ModelResponse",
    "PROVIDER_MAP",
]
