import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.yaml")

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary with the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            f"Please create it from 'config.yaml.template'."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    # Example of how to use the load_config function
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(config)
    except FileNotFoundError as e:
        print(e)
