import os
import json
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main(start_idx=0, end_idx=99, output_filename="spt_dataset.json"):
    # Load all 10 strategy prompts
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_files = [
        "1_analogy.txt",
        "2_compare_contrast.txt",
        "3_present_context.txt",
        "4_background_info.txt",
        "5_projection.txt",
        "6_reflection.txt",
        "7_stereotyping.txt",
        "8_emotion_regulation.txt",
        "9_information_extraction.txt",
        "10_open_mindedness.txt"
    ]

    # Load all prompt templates
    prompt_templates = {}
    for prompt_file in prompt_files:
        strategy_name = prompt_file.replace(".txt", "")
        prompt_path = prompts_dir / prompt_file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_templates[strategy_name] = f.read()
    print(f"✅ Loaded {len(prompt_templates)} strategy prompts")

    dataset = load_dataset("kellycyy/daily_dilemmas")
    print(f"✅ Loaded dataset")

    # Filter dataset by idx range
    print(f"🔍 Filtering idx range: {start_idx} - {end_idx}")

    all_results = []
    output_path = Path(__file__).parent / output_filename

    num_dilemmas = end_idx - start_idx + 1

    for dilemma_index in range(start_idx, end_idx + 1):
        print(f"\n🔄 Processing dilemma idx={dilemma_index} ({dilemma_index - start_idx + 1}/{num_dilemmas})...")
        dilemma = dataset['test'][dilemma_index]

        dilemma_result = {
            "dilemma_index": dilemma_index,
            "dilemma": dict(dilemma),
            "strategies": {}
        }

        # Process each strategy for this dilemma
        for strategy_name, prompt_template in prompt_templates.items():
            print(f"  🔄 Strategy: {strategy_name}...")

            # Inject dilemma into prompt
            final_prompt = prompt_template.format(
                dilemma=dilemma.get('dilemma_situation', 'N/A'),
                action=dilemma.get('action', 'N/A'),
                consequence=dilemma.get('negative_consequence', 'N/A')
            )

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[
                    {"role": "system", "content": "You are an expert in Social Perspective-Taking (SPT) dataset generation. Generate high-quality conversational data in Korean."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=4000
            )

            result = response.choices[0].message.content

            # Parse JSON: remove code blocks
            try:
                if result.startswith("```json"):
                    result = result.replace("```json", "").replace("```", "").strip()
                elif result.startswith("```"):
                    result = result.replace("```", "").strip()

                parsed_samples = json.loads(result)
                sample_count = len(parsed_samples.get("samples", []))
                print(f"  ✅ {strategy_name}: Generated {sample_count} samples")
            except json.JSONDecodeError as e:
                print(f"  ⚠️ {strategy_name}: JSON parsing failed: {e}")
                print(f"  Raw result: {result[:200]}...")
                parsed_samples = {"error": str(e), "raw": result}

            dilemma_result["strategies"][strategy_name] = parsed_samples

        all_results.append(dilemma_result)

        # Save after each dilemma
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved dilemma idx={dilemma_index} ({dilemma_index - start_idx + 1}/{num_dilemmas}) to {output_path}")

    print(f"\n✅ All done! Total {num_dilemmas} dilemmas x {len(prompt_templates)} strategies saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    # Generate dataset for idx 0-99 (100 dilemmas)
    main(start_idx=121, end_idx=150, output_filename="spt_dataset_121_150.json")
