# Jangmo Agent CoP Implementation Summary

## ✅ Completed Changes

### 1. File Updated
- **File**: `/Users/yoonnchaewon/Desktop/moral_agent/moral_agent_website/backend/agents/jangmo_agent/conversation_agent.py`
- **Before**: 725 lines
- **After**: 488 lines
- **Reduction**: 237 lines (33% reduction)

### 2. Removed Code

#### Classes & Constants
- ❌ `ResponseType` class
- ❌ `QuestionType` class
- ❌ `SPT_INDUCTION_QUESTIONS` list (8 questions)
- ❌ `EMPATHY_QUESTIONS` list (5 questions)
- ❌ `RESPONSE_TYPE_INSTRUCTIONS` dictionary
- ❌ `QUESTION_PATTERNS` class variable

#### Instance Variables
- ❌ `self.spt_state_store` - SPT state management
- ❌ `self.no_question_instruction` - SPT control

#### Methods Removed
- ❌ `get_spt_state()`
- ❌ `can_ask_question()`
- ❌ `increment_turn()`
- ❌ `get_spt_question_instruction()`
- ❌ `classify_response_type()`
- ❌ `_classify_by_rules()`
- ❌ `_classify_by_llm()`
- ❌ `get_dynamic_instruction()`
- ❌ `_load_no_question_instruction()`
- ❌ `_build_messages()` (old version with dynamic_instruction)

### 3. Added Code

#### Helper Methods (Character Definition)
```python
def _get_character_name(self) -> str:
    return "장모"

def _get_stance(self) -> str:
    return "AI 복원 반대"

def _get_speech_style(self) -> str:
    return "반말 (-냐, -지, -어)"

def _get_core_values(self) -> str:
    return "의무, 도리, 약속, 규범을 중시. 죽은 사람의 동의와 존엄성을 우선."

def _get_recent_history(self, messages: List[Dict[str, str]], limit: int = 3) -> str:
    # Returns recent N turns of conversation for SPT context

def _get_error_message(self) -> str:
    return "미안해, 다시 한번 말해줄래?"
```

#### CoP Core Methods
```python
async def _generate_self_reflection(self, messages, temperature=0.3) -> str:
    # Step 1 of 2-step CoP: Generate self-reflection with low temperature

def _build_reflection_prompt(self, messages) -> List:
    # Builds CoP + conditional SPT prompt with 4 steps:
    # Step 1: Character identity confirmation
    # Step 2: Dialogue context analysis + SPT necessity judgment (YES/NO)
    # Step 3: SPT 5 Q&A (only if YES)
    # Step 4: Response strategy decision

def _build_messages_with_reflection(self, messages, reflection) -> List:
    # Builds final response prompt with reflection result
```

#### Utility Method
```python
def _extract_chunk_text(self, chunk) -> str:
    # Extracts text from streaming chunks
```

### 4. Rewritten Methods

#### `chat()` Method
**Before**: ~60 lines with Response Classification and SPT state management
**After**: ~30 lines with 2-step CoP
```python
async def chat(self, messages, temperature=0.7, max_tokens=150, session_id="default"):
    # Step 1: Self-Reflection 생성
    reflection = await self._generate_self_reflection(messages, temperature=0.3)

    # Step 2: Reflection 기반으로 최종 응답 생성
    llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)
    lc_messages = self._build_messages_with_reflection(messages, reflection)
    result = await llm.ainvoke(lc_messages)

    # Step 3: Gemini 검증 (유지)
    cleaned_content = self.refine_response_with_gemini(...)

    # Step 4: 히스토리 저장
    history.add_user_message(last_user_msg)
    history.add_ai_message(cleaned_content)

    return cleaned_content
```

#### `chat_stream()` Method
**Before**: ~30 lines with Response Classification
**After**: ~25 lines with 2-step CoP (same structure as `chat()`)

### 5. Unchanged Code (Preserved)
- ✅ `session_store` - Conversation history management
- ✅ `get_session_history()` - Session retrieval
- ✅ `clear_session()` - Session cleanup
- ✅ `refine_response_with_gemini()` - Gemini validation (kept for double-checking persona)
- ✅ `_extract_last_user_message()` - Utility
- ✅ `_extract_previous_ai_messages()` - Utility
- ✅ `get_initial_message()` - Initial greeting
- ✅ `get_final_message()` - Farewell message
- ✅ `_create_llm()` - LLM instance creation

## 🔑 Key Changes

### From Complex Rule-Based to Simple LLM-Driven

**Before (Rule-Based)**:
```
User Input
  ↓
Response Type Classification (Rules + LLM)
  ↓
Get Dynamic Instruction (Based on ResponseType)
  ↓
SPT State Management (Turn counting, question selection)
  ↓
Get SPT Question Instruction (8 induction + 5 empathy questions)
  ↓
Build Messages (Complex prompt with dynamic instructions)
  ↓
Generate Response
  ↓
Gemini Validation
```

**After (CoP-Based)**:
```
User Input
  ↓
Self-Reflection (CoP + Conditional SPT)
  - Character identity confirmation
  - Context analysis
  - SPT necessity judgment (YES/NO)
  - If YES: 5 Q&A (contextual, not hardcoded)
  - Response strategy
  ↓
Generate Response (Based on reflection)
  ↓
Gemini Validation
```

### Conditional SPT Logic

**In Reflection Prompt**:
```
**Step 2: 대화 맥락 분석**
현재 대화 상황을 보고 다음을 판단하세요:
1. 사용자가 도덕적 딜레마에 대해 깊이 고민하고 있는가?
2. 다른 사람의 관점(이해관계자)을 고려해볼 필요가 있는가?
3. SPT를 통해 사용자의 사고를 확장할 수 있는가?

→ **SPT 필요성 판단: [YES/NO]**

**Step 3: SPT 자기 질문-답변 (필요한 경우에만)**
SPT가 필요하다고 판단되면 아래 5개 질문-답변을 수행하세요:

질문 1: 이 상황에서 가장 영향받는 이해관계자는 누구인가?
...
질문 5: 어떤 질문으로 사용자가 그 관점을 고려하도록 유도할 수 있는가?
```

## ✅ Verification

### Syntax Check
```bash
$ cd /Users/yoonnchaewon/Desktop/moral_agent/moral_agent_website/backend/agents/jangmo_agent
$ python3 -m py_compile conversation_agent.py
✅ Success (no output = no syntax errors)
```

### Import Test
```bash
$ python3 test_jangmo_cop.py
✅ Module loads successfully
✅ Agent instantiates correctly
⚠️ API key not configured (expected in test environment)
```

### Code References Check
```bash
$ grep -E "ResponseType|classify_response_type|increment_turn|get_dynamic_instruction|get_spt_question_instruction|spt_state_store" conversation_agent.py
✅ No matches found (all old code removed)
```

## 📊 Benefits

### Advantages
✅ **Code Complexity Reduced**: ~500 lines → ~250 lines (50% target), achieved 33% reduction (237 lines)
✅ **Maintainability**: Much simpler logic, easier to debug
✅ **Naturalness**: LLM autonomously handles dialogue flow
✅ **Persona Consistency**: Self-reflection ensures character awareness every turn
✅ **Extensibility**: Easy to add new agents (just implement helper methods)
✅ **Research Alignment**: Implements Chain of Persona methodology from paper
✅ **Conditional SPT**: Only performs SPT when contextually needed (performance optimization)
✅ **Contextual Questions**: SPT questions generated based on dialogue, not hardcoded

### Trade-offs
⚠️ **Response Time**: 1 LLM call → 2 LLM calls (but conditional SPT helps)
⚠️ **Cost**: Slight increase (reflection generation ~200 tokens)
⚠️ **Prompt Dependency**: Quality depends on reflection prompt design
⚠️ **SPT Judgment**: Relies on LLM to correctly judge SPT necessity

## 📝 Next Steps

### Phase 2: Apply to Other Agents
Following the same pattern for:
1. **Son Agent** (아들 - AI 복원 찬성, 존댓말)
   - Character: "아들"
   - Stance: "AI 복원 찬성"
   - Speech Style: "존댓말 (-요, -세요)"
   - Core Values: "결과, 행복, 위로를 중시. 가족이 느낄 행복과 실질적 도움을 우선."

2. **Colleague1 Agent** (동료 화가 - AI 예술 반대, 반말)
   - Character: "동료 화가 (50대 여성)"
   - Stance: "AI 예술 전시 반대"
   - Speech Style: "반말 (-냐, -지, -네)"
   - Core Values: "예술가의 의무와 책임, 고통과 성찰을 통한 창작, 인간만의 고유성 중시."

3. **Colleague2 Agent** (후배 화가 - AI 예술 찬성, 존댓말)
   - Character: "후배 화가 (30대 남성)"
   - Stance: "AI 예술 전시 찬성"
   - Speech Style: "존댓말 (-요, -세요)"
   - Core Values: "예술의 대중화, 감동과 즐거움, 새로운 가능성, 실용적 이익 중시."

### Testing Plan
1. Run existing tests: `test_all_agents.py`
2. Persona consistency check: `test_persona_evaluation.py`
3. 10-turn dialogue test: `test_colleague1_10turn.py`
4. Integration test: `test_persona_integration.py`

## 🎯 Success Criteria
- [ ] All 4 agents successfully converted to CoP
- [ ] Persona consistency maintained (말투, 입장, 가치관)
- [ ] Natural responses to various user input patterns
- [ ] Self-reflection quality verified (logs)
- [ ] Conditional SPT working correctly (YES/NO judgment)
- [ ] SPT 5 Q&A quality check (contextual relevance)
- [ ] Gemini validation still working
- [ ] All existing tests passing
- [ ] Response time acceptable (<5 seconds)
- [ ] Code complexity reduced across all agents

---

**Implementation Date**: 2026-01-18
**Status**: Jangmo Agent ✅ Complete | Phase 2 Pending
