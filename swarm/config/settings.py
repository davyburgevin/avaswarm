"""Centralised configuration using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project-swarm/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Active provider ---
    default_provider: Literal[
        "openai", "anthropic", "openrouter", "github_copilot", "vllm"
    ] = "openai"
    default_model: str = "gpt-4o"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Anthropic ---
    anthropic_api_key: str = ""

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- GitHub Copilot ---
    github_copilot_token: str = ""
    # --- GitHub OAuth (for device login flow) ---
    github_oauth_client_id: str = ""

    # --- vLLM ---
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "local-model"

    # --- Web gateway ---
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    web_secret_key: str = "change-me"

    # --- Email gateway ---
    email_host: str = ""
    email_imap_port: int = 993
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_address: str = ""
    email_password: str = ""
    email_check_interval: int = 60

    # --- Context window management ---
    # Token budget for conversation history (user-selectable: 4k/8k/16k/32k/64k/128k).
    # Converted to message count at runtime: max_messages = context_window // 200
    context_window: int = 16000

    # --- Memory ---
    memory_dir: Path = ROOT_DIR / "memory"
    long_term_file: str = "MEMORY.md"

    # --- Brave Search ---
    brave_search_api_key: str = ""

    # --- Scheduler ---
    scheduler_timezone: str = "Europe/Paris"

    # --- ClawHub ---
    clawhub_base_url: str = "https://clawhub.io/api/v1"
    clawhub_api_key: str = ""


# Singleton
settings = Settings()
