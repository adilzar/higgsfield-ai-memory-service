from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://memory:memory@db:5432/memory"
    database_url_sync: str = "postgresql://memory:memory@db:5432/memory"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek/deepseek-v4-flash"
    memory_auth_token: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


settings = Settings()
