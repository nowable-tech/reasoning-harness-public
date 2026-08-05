from .base import BaseAdapter, CredentialMissingError, ModelResponse
from .deepseek_adapter import DeepSeekAdapter
from .zai_adapter import ZaiAdapter
from .moonshot_adapter import MoonshotAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemma_adapter import GemmaAdapter
from .minimax_adapter import MinimaxAdapter
from .google_adapter import GoogleAdapter
from .mistral_adapter import MistralAdapter
from .thinkingmachines_adapter import ThinkingMachinesAdapter

PROVIDER_MAP: dict[str, type[BaseAdapter]] = {
    "deepseek": DeepSeekAdapter,
    "zai": ZaiAdapter,
    "moonshot": MoonshotAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemma": GemmaAdapter,       # Gemma 4 via OpenRouter
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
