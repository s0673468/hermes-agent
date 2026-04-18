"""Tests for empty model fallback — when provider is configured but model is missing."""

from unittest.mock import MagicMock, patch
import pytest


class TestGetDefaultModelForProvider:
    """Unit tests for hermes_cli.models.get_default_model_for_provider."""

    def test_known_provider_returns_first_model(self):
        from hermes_cli.models import get_default_model_for_provider
        result = get_default_model_for_provider("openai-codex")
        # Should return first model from _PROVIDER_MODELS["openai-codex"]
        assert result
        assert isinstance(result, str)

    def test_openrouter_returns_empty(self):
        """OpenRouter uses dynamic model fetch, no static catalog entry."""
        from hermes_cli.models import get_default_model_for_provider
        # OpenRouter is not in _PROVIDER_MODELS — it uses live fetching
        result = get_default_model_for_provider("openrouter")
        assert result == ""

    def test_unknown_provider_returns_empty(self):
        from hermes_cli.models import get_default_model_for_provider
        assert get_default_model_for_provider("nonexistent-provider") == ""

    def test_custom_provider_returns_empty(self):
        """Custom provider has no model catalog — should return empty."""
        from hermes_cli.models import get_default_model_for_provider
        # Custom providers don't have entries in _PROVIDER_MODELS
        assert get_default_model_for_provider("some-random-custom") == ""


class TestGatewayEmptyModelFallback:
    """Test that _resolve_session_agent_runtime fills in empty model from provider catalog."""

    def test_sticky_telegram_channel_model_binding_applies(self):
        """Per-channel Telegram model bindings should override the global default model."""
        from gateway.config import Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = MagicMock(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    extra={
                        "channel_model_bindings": {
                            "-5222898508": "lmstudio/qwen3.6-35b-a3b",
                        }
                    },
                )
            }
        )
        runner._session_key_for_source = lambda source: f"telegram:{source.chat_id}"

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-5222898508",
            chat_type="group",
            chat_name="Qwen Group",
        )

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "codex-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("hermes_cli.model_switch.switch_model") as switch_model_mock:
            switch_model_mock.return_value = MagicMock(
                success=True,
                new_model="qwen3.6-35b-a3b",
                target_provider="lmstudio",
                api_key="",
                base_url="http://192.168.15.9:1234/v1",
                api_mode="chat_completions",
            )

            model, kwargs = runner._resolve_session_agent_runtime(source=source)

        assert model == "qwen3.6-35b-a3b"
        assert kwargs["provider"] == "lmstudio"
        assert kwargs["base_url"] == "http://192.168.15.9:1234/v1"
        switch_model_mock.assert_called_once()

    def test_session_override_beats_sticky_channel_model_binding(self):
        """Explicit /model session override should take precedence over sticky channel binding."""
        from gateway.config import Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {
            "telegram:-5222898508": {
                "model": "gemma-4-26b-a4b-it",
                "provider": "lmstudio",
                "api_key": "",
                "base_url": "http://192.168.15.9:1234/v1",
                "api_mode": "chat_completions",
            }
        }
        runner.config = MagicMock(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    extra={
                        "channel_model_bindings": {
                            "-5222898508": "lmstudio/qwen3.6-35b-a3b",
                        }
                    },
                )
            }
        )
        runner._session_key_for_source = lambda source: f"telegram:{source.chat_id}"

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-5222898508",
            chat_type="group",
            chat_name="Qwen Group",
        )

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "codex-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime(source=source)

        assert model == "gemma-4-26b-a4b-it"
        assert kwargs["provider"] == "lmstudio"
        assert kwargs["base_url"] == "http://192.168.15.9:1234/v1"

    def test_empty_model_filled_from_provider(self):
        """When config has no model but provider is openai-codex, use first codex model."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        # Mock _resolve_gateway_model to return empty string
        # Mock _resolve_runtime_agent_kwargs to return openai-codex provider
        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Model should have been filled in from provider catalog
        assert model, "Model should not be empty when provider is known"
        assert isinstance(model, str)
        assert kwargs["provider"] == "openai-codex"

    def test_nonempty_model_not_overridden(self):
        """When config has a model set, don't override it."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        assert model == "gpt-5.4", "Explicit model should not be overridden"

    def test_empty_model_no_provider_stays_empty(self):
        """When both model and provider are empty, model stays empty."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "",
                 "api_key": "test-key",
                 "base_url": "https://example.com",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Can't fill in a default without knowing the provider
        assert model == ""


class TestResolveGatewayModel:
    """Test _resolve_gateway_model reads model from config correctly."""

    def test_returns_default_key(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"default": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_model_key_fallback(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"model": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_empty_when_missing(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {}}) == ""

    def test_returns_empty_when_no_model_section(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({}) == ""

    def test_string_model_config(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": "my-model"}) == "my-model"
