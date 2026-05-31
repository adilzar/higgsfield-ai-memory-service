from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Database
    db_host: str = "db"
    db_port: int = 5432
    db_name: str = "memory"
    db_user: str = "memory"
    db_password: str = "memory"

    # Override full URL if needed (takes precedence over individual fields)
    database_url: str = ""
    database_url_sync: str = ""

    # LLM
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek/deepseek-v4-flash"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Auth
    memory_auth_token: str = ""

    @computed_field
    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @computed_field
    @property
    def effective_database_url_sync(self) -> str:
        if self.database_url_sync:
            return self.database_url_sync
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
