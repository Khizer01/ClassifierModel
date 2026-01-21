from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
    database_path: str = str(_PROJECT_ROOT / "ads_data.db")
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    batch_size: int = 8
    max_length: int = 192
    max_new_tokens: int = 20
    device: str = "cpu"
    use_4bit_quantization: bool = False
    enable_tf32: bool = True
    enable_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        case_sensitive = False

    def model_post_init(self, __context) -> None:
        db_path = Path(self.database_path)
        if not db_path.is_absolute():
            self.database_path = str((_PROJECT_ROOT / db_path).resolve())

@lru_cache()
def get_settings() -> Settings:
    return Settings()
