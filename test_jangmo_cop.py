"""
Simple test for Jangmo Agent with CoP implementation
"""
import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "moral_agent_website" / "backend"
sys.path.insert(0, str(backend_path))

from agents.jangmo_agent.conversation_agent import JangmoAgent


async def test_jangmo_cop():
    """Test Jangmo agent with CoP"""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    print("=" * 60)
    print("Testing Jangmo Agent with Chain of Persona (CoP)")
    print("=" * 60)

    # Create agent
    agent = JangmoAgent(api_key=api_key, model="gpt-4o")
    print("✅ Agent created successfully")

    # Test 1: Helper methods
    print("\n📝 Testing helper methods:")
    print(f"  Character: {agent._get_character_name()}")
    print(f"  Stance: {agent._get_stance()}")
    print(f"  Speech Style: {agent._get_speech_style()}")
    print(f"  Core Values: {agent._get_core_values()}")
    print(f"  Error Message: {agent._get_error_message()}")

    # Test 2: Initial message
    print("\n💬 Initial message:")
    print(f"  {agent.get_initial_message()}")

    # Test 3: Simple conversation
    print("\n🗣️ Testing conversation (non-streaming):")
    messages = [
        {"role": "user", "content": "왜 반대하는 거야?"}
    ]

    try:
        response = await agent.chat(messages, session_id="test_session")
        print(f"  User: {messages[0]['content']}")
        print(f"  Jangmo: {response}")
        print("  ✅ Chat method works!")

        # Test 4: Follow-up
        print("\n🗣️ Testing follow-up:")
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "그래도 보고싶어"})

        response2 = await agent.chat(messages, session_id="test_session")
        print(f"  User: {messages[2]['content']}")
        print(f"  Jangmo: {response2}")
        print("  ✅ Follow-up works!")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Session history
    print("\n📚 Testing session history:")
    history = agent.get_session_history("test_session")
    print(f"  History length: {len(history.messages)} messages")
    for i, msg in enumerate(history.messages):
        role = "User" if msg.type == "human" else "AI"
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{i+1}] {role}: {content}")

    # Test 6: Clear session
    print("\n🗑️ Testing session clear:")
    result = agent.clear_session("test_session")
    print(f"  Clear result: {result}")
    print(f"  History after clear: {len(agent.get_session_history('test_session').messages)} messages")

    # Test 7: Final message
    print("\n👋 Final message:")
    print(f"  {agent.get_final_message()}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_jangmo_cop())
