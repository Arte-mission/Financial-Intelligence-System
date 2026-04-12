from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    NEWSDATA_API_KEY: str = "your_key_here" # Fallback if not configured properly, though relying on .env is ideal
    DATABASE_URL: str = "sqlite:///./market.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
