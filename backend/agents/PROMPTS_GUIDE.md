# Prompt Engineering Guide

This document describes the prompt design methodology used across the Moral Agent system. All agents use externalized prompt files loaded at runtime via `load_all_prompts()`.

## Agent Categories

### Facilitator Agents (Artist Apprentice, Friend)

These agents guide ethical dialogue without taking a stance. They use a **modular pipeline architecture** where each conversational action (intent detection, acknowledgment, question generation) has its own dedicated prompt.

**Pipeline per turn:**
1. Intent Detection - classify user's response type
2. Acknowledgment Generation - empathetic reflection of what user said
3. Action Selection - decide whether to explain, rephrase, transition, or explore
4. Response Generation - produce final output using the selected action prompt

### Persona Agents (Colleague 1/2, Jangmo, Son)

Fine-tuned models with a **two-phase reasoning architecture**:
1. **Reflection Phase** - internal reasoning with self-verification checklist
2. **Response Phase** - generate character-consistent output grounded in reflection

Optional **SPT Planning Phase** inserted between reflection and response when the reflection determines perspective-taking is needed.

---

## Prompt Design Patterns

### 1. Constraint-Based Generation

Most prompts use explicit constraint lists to control output behavior. Constraints are categorized as:
- **Forbidden phrases** - specific expressions the model must never use
- **Required elements** - components every response must include
- **Length limits** - word or sentence count caps
- **Style rules** - speech register, punctuation, tone

Example from `stage2_first_question.txt`:
```
Prohibited:
- Do NOT infer user's opinion (e.g., "you seem to think...")
- Do NOT use formal speech (no -yo, -sumnida endings)
- Do NOT exceed 2-3 sentences
```

### 2. Good/Bad Example Pairs

Generation prompts include paired examples showing correct vs. incorrect output. This technique is used heavily in acknowledgment and question generation prompts to prevent common failure modes like:
- Generic empathy ("that must be hard") instead of content-specific reflection
- Speculating about user's opinion instead of asking
- Repeating the same question verbatim

### 3. Few-Shot Classification

Intent detection prompts use extensive labeled examples (5-10 per category) with edge cases. Classification prompts output a single token label.

Categories vary by stage:
- **Stage 2**: `opinion`, `dont_know`, `unclear`
- **Stage 3**: `answer`, `need_reason`, `question`, `off_topic`, `ask_opinion`, `dont_understand`, `uncertain`, `closing`

### 4. Multi-Step Self-Verification

Persona agent reflection prompts use an 8-section checklist framework:
1. Character identity confirmation
2. Ethical stance alignment check
3. Dialogue principle review
4. Failure mode detection (Academic Critic, Aggressive Gatekeeper, Passive Observer, Repetitive Robot)
5. Previous question analysis (repetition prevention)
6. User utterance type classification
7. SPT necessity assessment (5-question YES/NO decision tree)
8. Context analysis output (JSON)

### 5. Perspective Rotation (SPT Planning)

The SPT planner prompt enforces stakeholder diversity across turns by requiring:
- Selection from predefined stakeholder categories (Artists, Apprentices, Audiences, Art Market, Tradition, Ethics)
- No category reuse across consecutive turns
- 5-step self-questioning: identify stakeholders, analyze emotions, check stance alignment, find blind spots, formulate strategic question

### 6. Verbatim Copying Requirement

Certain prompts (transition, guide-back, clarification) require the model to copy a provided question exactly without modification. This ensures conversation consistency when the system needs to re-ask a specific ethics question.

### 7. Anti-Repetition Mechanisms

Multiple layers prevent repetitive responses:
- **Variation libraries** in prompt files (6+ phrasings for common expressions)
- **Forbidden previous expressions** passed as context
- **Reflection-phase repetition checks** analyzing conversation history
- **Question variation sets** in `ethics_topics.json` (3 variations per topic)

---

## Prompt Inventory

### Facilitator Agents (Friend / Artist Apprentice)

Each has 18 prompt files + 1 JSON data file:

| File | Type | Purpose |
|---|---|---|
| `persona.txt` | System | Character definition, ethical framework mapping, conversation rules |
| `stage1_character_check.txt` | Classifier | Binary check if user provided concrete life details |
| `stage2_acknowledgment.txt` | Generator | Empathetic response reflecting user's specific words |
| `stage2_first_question.txt` | Generator | Initial ethics question with brief acknowledgment |
| `stage2_detect_intent.txt` | Classifier | Classify user stance (opinion / dont_know / unclear) |
| `stage2_detect_explanation.txt` | Classifier | Detect if user wants explanation (explain / continue) |
| `stage2_explanation.txt` | Generator | Explain the AI service concept without technical jargon |
| `stage2_rephrase.txt` | Generator | Rephrase question when user's answer is unclear |
| `stage2_to_stage3_transition.txt` | Generator | Bridge response transitioning to ethics exploration |
| `stage3_ask_opinion.txt` | Generator | Redirect when user asks for agent's opinion (express uncertainty) |
| `stage3_ask_why_unsure.txt` | Generator | Probe reasoning behind user's uncertainty |
| `stage3_clarification.txt` | Generator | Simplify question when user doesn't understand |
| `stage3_closing.txt` | Generator | Warm conversation closing |
| `stage3_concept_explanation.txt` | Generator | Define ethics concept in accessible language |
| `stage3_explain_reasoning.txt` | Generator | Explain reasoning through stakeholder perspectives |
| `stage3_guide_back.txt` | Generator | Redirect off-topic conversation to ethics framework |
| `stage3_request_reason.txt` | Generator | Ask user to explain reasoning behind stated opinion |
| `detect_intent.txt` | Classifier | 8-category intent classification for Stage 3 |
| `ai_opinion.txt` | Generator | Express uncertainty when asked for opinion |
| `ethics_topics.json` | Data | 5 ethics topics (beneficence, nonmaleficence, autonomy, justice, explicability) with 3 question variations each |

### Persona Agents (Colleague 1/2, Jangmo, Son)

Each has 3 prompt files:

| File | Type | Purpose |
|---|---|---|
| `reflection_prompt.txt` | Reasoning | Multi-step self-verification with failure mode detection |
| `spt_planner_prompt.txt` | Planning | Stakeholder perspective selection and strategic question design |
| `response_prompt.txt` | Generator | Character-consistent response with persona verification protocol |

### SPT Agent

| File | Type | Purpose |
|---|---|---|
| `system_prompt.txt` | System | 7 SPT strategies, empathetic listening rules, open-ended question requirements |
| `response_type_classifier.txt` | Classifier | 6-category attitude classification (AGREE, DISAGREE, UNCERTAIN, SHORT, QUESTION, OTHER) |

---

## Ethics Framework

All facilitator agents explore 5 ethical dimensions derived from Floridi's AI ethics framework:

1. **Beneficence** - Can this technology help people?
2. **Nonmaleficence** - Could this technology cause harm?
3. **Autonomy** - Does this affect people's ability to make their own choices?
4. **Justice** - Is the impact distributed fairly across groups?
5. **Explicability** - Can people understand how the technology works and why?

Each dimension has 3 question variations stored in `ethics_topics.json` to prevent repetitive questioning across sessions.
