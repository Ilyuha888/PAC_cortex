from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    benchmark_host: str = "https://api.bitgn.com"
    benchmark_id: str = "bitgn/pac1-dev"
    llm_model: str = "gpt-4.1-2025-04-14"
    llm_api_key: str = ""
    openai_base_url: str = ""
    vm_call_timeout_s: float = 10.0
    vm_call_retries: int = 2
    api_call_budget: int = 1000


settings = Settings()
