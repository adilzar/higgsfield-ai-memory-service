from src.config import Settings


def test_settings_load_from_env_file():
    assert Settings.model_config["env_file"] == ".env"
