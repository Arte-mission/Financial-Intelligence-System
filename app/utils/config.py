from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Required for Gemini AI interpretation layer
    GEMINI_API_KEY: Optional[str] = None

    # Optional — used as a fallback news source if OnlineKhabar yields too few articles.
    # App runs normally without this key (OnlineKhabar-only mode).
    NEWSDATA_API_KEY: Optional[str] = None

    DATABASE_URL: str = "sqlite:///./market.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
