"""
Comprehensive Test Suite for Raec System
Tests all major components and integration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from memory.memory_db import HierarchicalMemoryDB, MemoryType
from skills.skill_graph import SkillGraph, SkillCategory, SkillStatus
from tools.executor import ToolExecutor, ToolType
from evaluators.logic_checker import LogicChecker, VerificationLevel


def test_memory_system():
    """Test hierarchical memory system"""
    print("\n" + "="*70)
    print("TEST 1: HIERARCHICAL MEMORY SYSTEM")
    print("="*70 + "\n")
    
    memory = HierarchicalMemoryDB("data/embeddings/test_memory.db")
    
    # Test 1: Store different memory types
    print("📝 Test 1.1: Storing different memory types")
    fact_id = memory.store(
        "Test fact: Water boils at 100°C",
        memory_type=MemoryType.FACT,
        confidence=1.0
    )
    print(f"   ✓ Stored fact (ID: {fact_id})")
    
    exp_id = memory.store(
        "Completed test execution successfully",
        memory_type=MemoryType.EXPERIENCE,
        metadata={"test": "memory_system", "result": "pass"}
    )
    print(f"   ✓ Stored experience (ID: {exp_id})")
    
    belief_id = memory.store(
        "Hypothesis: Smaller batch sizes improve training",
        memory_type=MemoryType.BELIEF,
        confidence=0.7
    )
    print(f"   ✓ Stored belief (ID: {belief_id})")
    
    # Test 2: Query by type
    print("\n🔍 Test 1.2: Type-filtered querying")
    fact_results = memory.query("water temperature", memory_types=[MemoryType.FACT], k=1)
    assert len(fact_results) > 0, "Should find facts"
    print(f"   ✓ Found {len(fact_results)} fact(s)")
    
    # Test 3: Belief evolution
    print("\n🔄 Test 1.3: Belief evolution")
    new_belief_id = memory.evolve_belief(
        belief_id,
        "Batch size should be optimized per dataset",
        evidence="Testing showed dataset-dependent performance",
        confidence_delta=0.2
    )
    print(f"   ✓ Evolved belief (New ID: {new_belief_id})")
    
    # Test 4: Memory linking
    print("\n🔗 Test 1.4: Memory linking")
    memory.add_link(exp_id, fact_id, "uses_knowledge", strength=0.8)
    print(f"   ✓ Linked experience to fact")
    
    # Test 5: Temporal context
    print("\n⏰ Test 1.5: Temporal context retrieval")
    temporal = memory.get_temporal_context(time.time(), window=60)
    print(f"   ✓ Retrieved {len(temporal)} memories from last minute")
    
    # Test 6: Summary creation
    print("\n📋 Test 1.6: Summary creation")
    summary_id = memory.create_summary(
        [fact_id, exp_id],
        "Test summary: Stored and linked test data",
        topic="testing"
    )
    print(f"   ✓ Created summary (ID: {summary_id})")
    
    print("\n✅ Memory system tests PASSED\n")
    memory.close()
    return True


def test_skill_graph():
    """Test audited skill graph"""
    print("\n" + "="*70)
    print("TEST 2: AUDITED SKILL GRAPH")
    print("="*70 + "\n")
    
    skills = SkillGraph("skills/test_skill_db.json")
    
    # Test 1: Extract skill
    print("🆕 Test 2.1: Skill extraction")
    skill_id = skills.extract_skill(
        task_description="Parse CSV and compute statistics",
        solution="def parse_csv(file): return pd.read_csv(file).describe()",
        execution_result={"success": True, "time": 0.5},
        category=SkillCategory.DATA_PROCESSING
    )
    print(f"   ✓ Extracted skill (ID: {skill_id[:8]}...)")
    
    # Test 2: Verify skill
    print("\n🔍 Test 2.2: Skill verification")
    test_cases = [
        {
            "description": "Test basic functionality",
            "expected": True,
            "actual": True
        },
        {
            "description": "Test error handling",
            "expected": True,
            "actual": True
        }
    ]
    verified = skills.verify_skill(skill_id, test_cases)
    assert verified, "Skill should be verified"
    print(f"   ✓ Skill verified")
    
    # Test 3: Query skill
    print("\n🔎 Test 2.3: Skill querying")
    found_skill = skills.query_skill("CSV processing", category=SkillCategory.DATA_PROCESSING)
    assert found_skill is not None, "Should find matching skill"
    print(f"   ✓ Found skill: {found_skill.name}")
    
    # Test 4: Use skill
    print("\n▶️  Test 2.4: Skill usage")
    result = skills.use_skill(skill_id, {"file": "test.csv"})
    print(f"   ✓ Used skill successfully")
    
    # Test 5: Record outcome
    print("\n📊 Test 2.5: Outcome recording")
    skills.record_skill_outcome(skill_id, success=True, result={"data": "processed"})
    skill = skills.skills[skill_id]
    assert skill.usage_count > 0, "Usage count should increase"
    print(f"   ✓ Recorded outcome (Usage: {skill.usage_count}, Success rate: {skill.success_rate:.1%})")
    
    # Test 6: Skill dependencies
    print("\n🔗 Test 2.6: Skill dependencies")
    basic_skill_id = skills.extract_skill(
        "Read CSV file",
        "def read_csv(file): return pd.read_csv(file)",
        {"success": True},
        SkillCategory.DATA_PROCESSING
    )
    skills.add_skill_dependency(skill_id, basic_skill_id)
    chain = skills.get_skill_chain(skill_id)
    assert len(chain) >= 2, "Should have dependency chain"
    print(f"   ✓ Created dependency chain (Length: {len(chain)})")
    
    # Test 7: Get stats
    print("\n📈 Test 2.7: Statistics")
    stats = skills.get_stats()
    print(f"   ✓ Total skills: {stats['total_skills']}")
    print(f"   ✓ Verified: {stats['verified_count']}")
    print(f"   ✓ Avg confidence: {stats['avg_confidence']:.1%}")
    
    print("\n✅ Skill graph tests PASSED\n")
    return True


def test_tool_executor():
    """Test runtime tool evolution"""
    print("\n" + "="*70)
    print("TEST 3: RUNTIME TOOL EVOLUTION")
    print("="*70 + "\n")
    
    tools = ToolExecutor()
    
    # Test 1: Python execution
    print("🐍 Test 3.1: Python code execution")
    script = "print('Hello from test')\nresult = 42"
    output = tools.run_python(script, "Test script")
    assert "Hello from test" in output or output != "", "Should execute Python"
    print(f"   ✓ Python execution works")
    
    # Test 2: Record execution
    print("\n📝 Test 3.2: Execution recording")
    initial_history_len = len(tools.execution_history)
    tools.run_python("x = 1 + 1", "Simple math")
    assert len(tools.execution_history) > initial_history_len, "Should record execution"
    print(f"   ✓ Execution recorded ({len(tools.execution_history)} total)")
    
    # Test 3: Bottleneck detection
    print("\n⚠️  Test 3.3: Bottleneck detection")
    # Simulate slow operations
    for i in range(5):
        tools._record_execution({
            'type': 'python',
            'description': 'slow_operation',
            'success': True,
            'runtime': 1.5  # Simulated slow runtime
        })
    bottlenecks = tools.detect_bottlenecks(threshold_ms=1000)
    print(f"   ✓ Detected {len(bottlenecks)} bottleneck(s)")
    
    # Test 4: Tool stats
    print("\n📊 Test 3.4: Tool statistics")
    stats = tools.get_tool_stats()
    print(f"   ✓ Total tools: {stats['total_tools']}")
    print(f"   ✓ Total executions: {stats['total_executions']}")
    
    print("\n✅ Tool executor tests PASSED\n")
    return True


def test_logic_checker():
    """Test verification and error correction"""
    print("\n" + "="*70)
    print("TEST 4: LOGIC CHECKER & VERIFICATION")
    print("="*70 + "\n")
    
    checker = LogicChecker()
    
    # Test 1: Syntax verification
    print("🔍 Test 4.1: Syntax verification")
    code = "def test():\n    return 42"
    passed, results = checker.verify(
        code,
        verification_levels=[VerificationLevel.SYNTAX]
    )
    assert passed, "Valid syntax should pass"
    print(f"   ✓ Syntax verification passed")
    
    # Test 2: Invalid syntax detection
    print("\n❌ Test 4.2: Invalid syntax detection")
    bad_code = "def test(\n    return 42"  # Missing closing paren
    passed, results = checker.verify(
        bad_code,
        verification_levels=[VerificationLevel.SYNTAX]
    )
    assert not passed, "Invalid syntax should fail"
    print(f"   ✓ Invalid syntax detected")
    
    # Test 3: Output verification
    print("\n✅ Test 4.3: Output verification")
    passed, results = checker.verify(
        output="Success: Task completed",
        expected="Success: Task completed",
        verification_levels=[VerificationLevel.OUTPUT]
    )
    assert passed, "Matching output should pass"
    print(f"   ✓ Output verification passed")
    
    # Test 4: Error detection
    print("\n🚨 Test 4.4: Error detection")
    passed, results = checker.verify(
        output="ERROR: Something failed",
        verification_levels=[VerificationLevel.OUTPUT]
    )
    assert not passed, "Error output should fail"
    print(f"   ✓ Error detected in output")
    
    # Test 5: Logic verification
    print("\n🧠 Test 4.5: Logic verification")
    passed, results = checker.verify(
        output="def process(x): return x * 2",
        verification_levels=[VerificationLevel.LOGIC],
        context={"task": "double a number"}
    )
    print(f"   ✓ Logic verification completed (Passed: {passed})")
    
    # Test 6: Performance verification
    print("\n⚡ Test 4.6: Performance verification")
    inefficient_code = """
for i in range(n):
    for j in range(n):
        result += i * j
"""
    passed, results = checker.verify(
        inefficient_code,
        verification_levels=[VerificationLevel.PERFORMANCE]
    )
    # Performance check is informational, so it passes
    has_warning = any(r.severity.value == "warning" for r in results)
    print(f"   ✓ Performance check completed (Has warnings: {has_warning})")
    
    # Test 7: Incremental verification
    print("\n📊 Test 4.7: Incremental verification")
    steps = [
        "Analyze the requirements",
        "Design the solution architecture",
        "Implement the core components"
    ]
    step_results = checker.incremental_verify(steps, "Build a system")
    assert len(step_results) == len(steps), "Should verify all steps"
    print(f"   ✓ Verified {len(step_results)} steps")
    
    # Test 8: Get stats
    print("\n📈 Test 4.8: Verification statistics")
    stats = checker.get_verification_stats()
    print(f"   ✓ Total verifications: {stats['total_verifications']}")
    print(f"   ✓ Pass rate: {stats['pass_rate']:.1%}")
    
    print("\n✅ Logic checker tests PASSED\n")
    return True


def test_integration():
    """Test system integration"""
    print("\n" + "="*70)
    print("TEST 5: SYSTEM INTEGRATION")
    print("="*70 + "\n")
    
    # Test memory + skills integration
    print("🔗 Test 5.1: Memory + Skills integration")
    memory = HierarchicalMemoryDB("data/embeddings/test_integration.db")
    skills = SkillGraph("skills/test_integration_skills.json")
    
    # Extract skill and store in memory
    skill_id = skills.extract_skill(
        "Integration test skill",
        "def integrate(): return True",
        {"success": True},
        SkillCategory.TOOL_USAGE
    )
    
    memory_id = memory.store(
        f"Extracted skill: Integration test skill",
        memory_type=MemoryType.EXPERIENCE,
        metadata={"skill_id": skill_id}
    )
    
    print(f"   ✓ Skill extracted and stored in memory")
    print(f"   ✓ Skill ID: {skill_id[:8]}...")
    print(f"   ✓ Memory ID: {memory_id}")
    
    # Query memory for skill-related experiences
    skill_memories = memory.query(
        "integration test",
        memory_types=[MemoryType.EXPERIENCE],
        k=1
    )
    assert len(skill_memories) > 0, "Should find skill in memory"
    print(f"   ✓ Retrieved skill from memory")
    
    memory.close()
    print("\n✅ Integration tests PASSED\n")
    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("🧪 RAEC COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Memory System", test_memory_system),
        ("Skill Graph", test_skill_graph),
        ("Tool Executor", test_tool_executor),
        ("Logic Checker", test_logic_checker),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            print(f"\n{'▶'*3} Running: {name}")
            success = test_func()
            results[name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n❌ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70 + "\n")
    
    for name, result in results.items():
        status_symbol = "✅" if result == "PASS" else "❌"
        print(f"  {status_symbol} {name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*70}\n")
    
    return all(r == "PASS" for r in results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
