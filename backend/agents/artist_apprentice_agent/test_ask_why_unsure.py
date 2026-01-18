"""
화가지망생 에이전트의 "왜 그렇게 생각하는지" 질문 기능 테스트
"""
import sys
from pathlib import Path
import os

# 경로 설정
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

from langchain_openai import ChatOpenAI

# 직접 패턴 테스트
_DONT_KNOW_PATTERNS = (
    "모르겠어",
    "모르겟어",  # 오타 변형
    "모르겠어요",
    "잘 모르겠어",
    "잘모르겠어",
    "잘 모르겠어요",
    "잘모르겠어요",
    "글쎄",
    "글세",  # 오타 변형
    "글쎄요",
    "몰라",
    "모르겠는데",
    "모르겟는데",  # 오타 변형
    "생각이 안 나",
    "생각안나",
    "어?",  # 짧은 혼란 표현
    "?",  # 단독 물음표
)

def test_pattern_matching():
    """패턴 매칭 테스트 (휴리스틱)"""
    print("=" * 60)
    print("화가지망생 에이전트 - ask_why_unsure 패턴 매칭 테스트")
    print("=" * 60)
    
    # 테스트 케이스
    test_cases = [
        ("모르겠어", True),
        ("모르겠어요", True),
        ("잘 모르겠어", True),
        ("글쎄", True),
        ("글쎄요", True),
        ("어?", True),
        ("?", True),
        ("생각이 안 나", True),
        ("이 기술은 사회 전체에 공정하다고 보시나요?", False),  # 일반 질문
        ("좋을 것 같아요", False),  # 일반 답변
        ("응", False),  # 단답
        ("그래요", False),  # 동의
    ]
    
    print("\n패턴 매칭 테스트:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for user_message, should_match in test_cases:
        normalized = user_message.replace(" ", "").lower()
        user_stripped = user_message.strip()
        matched = False
        matched_pattern = None
        
        # 단독 물음표 체크 (가장 먼저)
        if user_stripped == "?":
            matched = True
            matched_pattern = "?"
        else:
            # 패턴 체크 ("?"는 이미 위에서 체크했으므로 제외)
            for pattern in _DONT_KNOW_PATTERNS:
                if pattern == "?":  # 단독 물음표는 이미 체크했으므로 건너뛰기
                    continue
                    
                pattern_normalized = pattern.replace(" ", "").lower()
                
                # 정확히 일치하는 경우
                if user_stripped == pattern.strip():
                    matched = True
                    matched_pattern = pattern
                    break
                
                # "어?" 패턴은 문장 시작 부분에만 매칭 (긴 문장 제외)
                if pattern == "어?":
                    if user_stripped.startswith("어?") and len(user_stripped) <= 5:
                        matched = True
                        matched_pattern = pattern
                        break
                else:
                    # 다른 패턴은 포함 여부로 체크
                    if pattern_normalized in normalized:
                        matched = True
                        matched_pattern = pattern
                        break
        
        if matched == should_match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        pattern_info = f" (패턴: {matched_pattern})" if matched_pattern else ""
        print(f"{status}: '{user_message}' → matched={matched} (예상: {should_match}){pattern_info}")
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 모든 패턴 매칭 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")
    
    # 프롬프트 파일 확인
    print("\n" + "=" * 60)
    print("프롬프트 파일 확인")
    print("=" * 60)
    
    prompts_path = Path(__file__).parent / "prompts" / "stage3_ask_why_unsure.txt"
    if prompts_path.exists():
        print(f"✅ 프롬프트 파일 존재: {prompts_path}")
        with open(prompts_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"   파일 크기: {len(content)} bytes")
            print(f"   내용 미리보기:\n{content[:200]}...")
    else:
        print(f"❌ 프롬프트 파일 없음: {prompts_path}")

if __name__ == "__main__":
    test_pattern_matching()

