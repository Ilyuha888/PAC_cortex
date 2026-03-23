from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    bitgn_api_key: str = ""
    bitgn_api_url: str = "https://api.bitgn.com"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    api_call_budget: int = 1000


settings = Settings()
