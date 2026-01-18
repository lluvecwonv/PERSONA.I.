"""
유틸리티 함수 모음
"""
import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str, base_path: str = None) -> str:
    """
    프롬프트 파일을 로드합니다.

    Args:
        prompt_name: 프롬프트 파일 이름 (확장자 제외)
        base_path: 프롬프트 파일이 있는 디렉토리 경로 (기본값: artist_apprentice_agent/prompts/)

    Returns:
        프롬프트 텍스트

    Raises:
        FileNotFoundError: 프롬프트 파일이 없을 경우
    """
    if base_path is None:
        # 기본 경로: backend/artist_apprentice_agent/prompts/
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
    """
    모든 프롬프트 파일을 딕셔너리로 로드합니다.

    Args:
        base_path: 프롬프트 파일이 있는 디렉토리 경로

    Returns:
        {프롬프트명: 프롬프트 텍스트} 딕셔너리
    """
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
    """
    프롬프트 템플릿에 변수를 채웁니다.
    .replace()를 사용하여 지정된 키만 치환하고, 다른 중괄호는 그대로 유지합니다.

    Args:
        prompt_template: 프롬프트 템플릿 문자열
        **kwargs: 템플릿에 채울 변수들

    Returns:
        포맷된 프롬프트

    Example:
        >>> template = "안녕하세요 {name}님, {greeting}"
        >>> format_prompt(template, name="홍길동")
        '안녕하세요 홍길동님, {greeting}'  # 지정되지 않은 변수는 그대로 유지
    """
    result = prompt_template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def load_json_data(file_name: str, base_path: str = None) -> Dict[str, Any]:
    """
    JSON 파일을 로드합니다.

    Args:
        file_name: JSON 파일 이름 (확장자 포함)
        base_path: JSON 파일이 있는 디렉토리 경로 (기본값: artist_apprentice_agent/prompts/)

    Returns:
        JSON 데이터 딕셔너리

    Raises:
        FileNotFoundError: JSON 파일이 없을 경우
    """
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