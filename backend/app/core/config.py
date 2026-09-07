from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AIDA - Autonomous Intelligence & Data Analyst"
    API_V1_STR: str = "/api/v1"
    RANDOM_SEED: int = 42

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")


settings = Settings()
