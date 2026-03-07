"""Utility functions."""
import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str, base_path: str = None) -> str:
    if base_path is None:
        current_file = Path(__file__)
        base_path = current_file.parent / "artist_apprentice_agent" / "prompts"
    else:
        base_path = Path(base_path)

    prompt_file = base_path / f"{prompt_name}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    logger.debug(f"Loaded prompt: {prompt_name} from {prompt_file}")
    return content


def load_all_prompts(base_path: str = None) -> Dict[str, str]:
    if base_path is None:
        current_file = Path(__file__)
        base_path = current_file.parent / "artist_apprentice_agent" / "prompts"
    else:
        base_path = Path(base_path)

    prompts = {}

    if not base_path.exists():
        logger.warning(f"Prompts directory not found: {base_path}")
        return prompts

    for prompt_file in base_path.glob("*.txt"):
        prompt_name = prompt_file.stem
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read()
            logger.debug(f"Loaded prompt: {prompt_name}")
        except Exception as e:
            logger.error(f"Failed to load prompt {prompt_name}: {e}")

    logger.info(f"Loaded {len(prompts)} prompts from {base_path}")
    return prompts


def format_prompt(prompt_template: str, **kwargs) -> str:
    """Replace only specified keys, leaving other braces intact."""
    result = prompt_template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def load_json_data(file_name: str, base_path: str = None) -> Dict[str, Any]:
    if base_path is None:
        current_file = Path(__file__)
        base_path = current_file.parent / "artist_apprentice_agent" / "prompts"
    else:
        base_path = Path(base_path)

    json_file = base_path / file_name

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON: {file_name} from {json_file}")
    return data
