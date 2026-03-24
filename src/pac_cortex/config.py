from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    benchmark_host: str = "https://api.bitgn.com"
    benchmark_id: str = "bitgn/pac1-dev"
    llm_model: str = "gpt-4.1-2025-04-14"
    llm_api_key: str = ""


settings = Settings()
