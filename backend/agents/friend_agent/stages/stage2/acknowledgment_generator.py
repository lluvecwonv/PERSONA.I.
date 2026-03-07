"""Generate empathy response for Stage 1 -> Stage 2 transition."""
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt


class AcknowledgmentGenerator:
    # Fixed Stage 2 question (exact wording, never change)
    STAGE2_QUESTION = "내일 그… 아내분 기일이잖아. 그냥 네가 걱정돼서 와봤어. 그, 왜 AI로 죽은 사람을 다시 재현한다는 기술 구매했다는 사람들이 꽤 있잖아. 넌 어떻게 생각해?"

    @staticmethod
    def generate(llm: ChatOpenAI, messages: list, prompts: dict) -> str:
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"] if user_messages else ""

        recent_messages = messages[-6:] if len(messages) > 6 else messages
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages
        ])

        acknowledgment_template = prompts.get(
            "stage2_acknowledgment",
            "친구의 말에 대해 짧게 공감하세요. 친구의 말: {user_message}"
        )

        acknowledgment_prompt = acknowledgment_template.replace("{user_message}", user_message)
        acknowledgment_prompt = acknowledgment_prompt.replace("{conversation_history}", conversation_history)

        try:
            result = llm.invoke(acknowledgment_prompt)
            acknowledgment = result.content.strip().strip('"')

            # Deduplicate: ensure Stage 2 question appears exactly once
            stage2_q = AcknowledgmentGenerator.STAGE2_QUESTION

            partial_matches = ["넌 어떻게 생각해?", "아내분 기일", "AI로 죽은 사람"]
            has_partial_match = any(p in acknowledgment for p in partial_matches)

            count = acknowledgment.count(stage2_q)

            if count > 0:
                parts = acknowledgment.split(stage2_q)
                clean_acknowledgment = parts[0].strip()

                if clean_acknowledgment:
                    return f"{clean_acknowledgment} {stage2_q}"
                else:
                    return stage2_q

            if has_partial_match:
                return acknowledgment

            return f"{acknowledgment} {stage2_q}"
        except Exception:
            return AcknowledgmentGenerator.STAGE2_QUESTION
