# Prompt Engineering Guide

This document describes the prompt design methodology used across the Moral Agent system. All agents use externalized prompt files loaded at runtime.

## Agent Categories

### Facilitator Agent (Artist Apprentice, Friend)

A single unified `FacilitatorAgent` class parameterized by `persona_type` ("artist_apprentice" | "friend"). Guides ethical dialogue without taking a stance, using a **3-phase architecture**:

**Phase 1: Self-Reflection** (cheap/fast model - Gemini Flash)
- Colleague1-style multi-step self-reflection
- 5 steps: character identity, dialogue principles check, previous question analysis, stage/intent classification, SPT necessity check
- Outputs JSON: `{ stage, action, intent, spt_required, context_analysis, forbidden_questions, suggested_questions }`

**Phase 1.5: SPT Planning** (cheap/fast model - conditional)
- Only invoked when Phase 1 determines `spt_required: true`
- Stakeholder perspective rotation across 5 categories
- 5-step self-questioning: identify stakeholders, analyze emotions, find blind spots, design strategic question
- Outputs JSON: `{ stakeholders, empathy_analysis, blind_spot, strategic_question }`

**Phase 2: Response Generation** (main model - GPT-4o)
- Self-reflection checks before output (persona, relationship, speech style, neutrality, repetition)
- Intent-specific response rules built into the prompt
- Question variation enforcement

**3-Stage Conversation Flow:**
1. **Stage 1** - Character/situation setup (casual greetings, life description)
2. **Stage 2** - AI stance question (introduce the ethical dilemma)
3. **Stage 3** - Ethics question loop (5 Floridi values: beneficence, nonmaleficence, autonomy, justice, explicability)

### Persona Agents (Colleague 1/2, Jangmo, Son)

Models with a **two-phase reasoning architecture**:
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

### 2. Good/Bad Example Pairs

Generation prompts include paired examples showing correct vs. incorrect output. This prevents common failure modes like:
- Generic empathy instead of content-specific reflection
- Speculating about user's opinion instead of asking
- Repeating the same question verbatim

### 3. Multi-Step Self-Verification

Reflection prompts use a structured checklist framework:
1. Character identity confirmation
2. Dialogue principle compliance check
3. Previous question analysis (repetition prevention)
4. Stage determination + intent classification
5. SPT necessity assessment (5-question YES/NO decision tree)
6. Context analysis output (JSON)

### 4. Perspective Rotation (SPT Planning)

The SPT planner prompt enforces stakeholder diversity across turns by requiring:
- Selection from predefined stakeholder categories (Beneficiaries, Affected parties, Autonomy holders, Justice seekers, Accountability holders)
- No category reuse across consecutive turns
- 5-step self-questioning: identify stakeholders, analyze emotions, find blind spots, formulate strategic question

### 5. Verbatim Copying Requirement

Certain response rules (advance_question, clarification) require the model to copy a provided question exactly without modification. This ensures conversation consistency when the system needs to ask a specific ethics question.

### 6. Anti-Repetition Mechanisms

Multiple layers prevent repetitive responses:
- **Forbidden questions list** passed from reflection phase
- **Variation libraries** in prompt files (6+ phrasings for common expressions)
- **Reflection-phase repetition checks** analyzing conversation history
- **Question variation sets** in `ethics_topics.json` (3 variations per topic)

---

## Prompt Inventory

### Facilitator Agent (Friend / Artist Apprentice)

Shared prompts (3 files):

| File | Type | Purpose |
|---|---|---|
| `reflection_prompt.txt` | Reasoning | 5-step self-reflection: identity, principles, repetition check, intent classification, SPT check |
| `spt_planner_prompt.txt` | Planning | Stakeholder perspective rotation and strategic question design |
| `response_prompt.txt` | Generator | Self-reflection checks + intent-specific response rules + question variation |

Per-persona assets (3 files each):

| File | Type | Purpose |
|---|---|---|
| `persona.txt` | System | Character definition, ethical framework, conversation rules, speech style |
| `config.json` | Data | Greetings, stage questions, fallback messages (7 keys) |
| `ethics_topics.json` | Data | 5 ethics topics with 3 question variations each |

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
