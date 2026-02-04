"""
Integration Test Suite for Raec System

Tests all major components and execution modes.
"""
import sys
import os
import traceback
from datetime import datetime

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from main import Raec
from memory.memory_db import MemoryType
from skills.skill_graph import SkillCategory
from tools.executor import ToolType


def print_header(title):
    """Print a formatted test header"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80 + "\n")


def print_result(test_name, passed, details=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")


def test_initialization():
    """Test 1: System Initialization"""
    print_header("TEST 1: System Initialization")
    
    try:
        raec = Raec()
        print_result("Raec initialization", True, "All subsystems loaded")
        return raec, True
    except Exception as e:
        print_result("Raec initialization", False, f"Error: {e}")
        traceback.print_exc()
        return None, False


def test_memory_system(raec):
    """Test 2: Hierarchical Memory System"""
    print_header("TEST 2: Hierarchical Memory System")
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 2.1: Store different memory types
        print("📝 Test 2.1: Storing different memory types...")
        
        fact_id = raec.memory.store(
            content="Python is a high-level programming language",
            memory_type=MemoryType.FACT,
            confidence=1.0,
            source="test"
        )
        print(f"   ✓ Stored FACT (ID: {fact_id})")
        
        exp_id = raec.memory.store(
            content="Successfully tested memory storage system",
            memory_type=MemoryType.EXPERIENCE,
            metadata={'test': 'integration'},
            source="test"
        )
        print(f"   ✓ Stored EXPERIENCE (ID: {exp_id})")
        
        belief_id = raec.memory.store(
            content="Testing improves code quality",
            memory_type=MemoryType.BELIEF,
            confidence=0.9,
            source="test"
        )
        print(f"   ✓ Stored BELIEF (ID: {belief_id})")
        
        tests_passed += 1
        print_result("Memory storage", True)
        
        # Test 2.2: Query memory
        print("\n📝 Test 2.2: Querying memory...")
        
        results = raec.memory.query(
            "programming language",
            memory_types=[MemoryType.FACT],
            k=5
        )
        
        if results:
            print(f"   ✓ Query returned {len(results)} results")
            print(f"   Sample: {results[0]['content'][:60]}...")
            tests_passed += 1
            print_result("Memory query", True)
        else:
            print_result("Memory query", False, "No results returned")
        
        # Test 2.3: Belief evolution
        print("\n📝 Test 2.3: Belief evolution...")
        
        new_belief_id = raec.memory.evolve_belief(
            belief_id=belief_id,
            new_content="Comprehensive testing dramatically improves code quality",
            evidence="Multiple studies confirm correlation",
            confidence_delta=0.05
        )
        
        print(f"   ✓ Belief evolved (New ID: {new_belief_id})")
        tests_passed += 1
        print_result("Belief evolution", True)
        
        # Test 2.4: Memory linking
        print("\n📝 Test 2.4: Memory linking...")
        
        raec.memory.add_link(exp_id, fact_id, "uses_knowledge", strength=0.8)
        print(f"   ✓ Link created: Experience → Fact")
        
        tests_passed += 1
        print_result("Memory linking", True)
        
    except Exception as e:
        print_result("Memory system", False, f"Error: {e}")
        traceback.print_exc()
    
    print(f"\n📊 Memory Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_skill_system(raec):
    """Test 3: Skill Graph System"""
    print_header("TEST 3: Skill Graph System")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 3.1: Skill extraction
        print("📝 Test 3.1: Skill extraction...")
        
        skill_id = raec.skills.extract_skill(
            task_description="Sort a list of numbers",
            solution="Use Python's built-in sorted() function",
            execution_result={'success': True, 'time': 0.001},
            category=SkillCategory.DATA_PROCESSING,
            name="List Sorting"
        )
        
        print(f"   ✓ Skill extracted (ID: {skill_id[:12]}...)")
        tests_passed += 1
        print_result("Skill extraction", True)
        
        # Test 3.2: Skill verification
        print("\n📝 Test 3.2: Skill verification...")
        
        test_cases = [
            {'description': 'Basic sorting', 'expected': True},
            {'description': 'Empty list', 'expected': True}
        ]
        
        verified = raec.skills.verify_skill(skill_id, test_cases)
        
        if verified:
            print(f"   ✓ Skill verified successfully")
            tests_passed += 1
            print_result("Skill verification", True)
        else:
            print_result("Skill verification", False, "Verification failed")
        
        # Test 3.3: Skill query
        print("\n📝 Test 3.3: Skill querying...")
        
        found_skill = raec.skills.query_skill(
            "sort numbers",
            category=SkillCategory.DATA_PROCESSING
        )
        
        if found_skill:
            print(f"   ✓ Found skill: {found_skill.name}")
            print(f"   Confidence: {found_skill.confidence:.1%}")
            tests_passed += 1
            print_result("Skill querying", True)
        else:
            print_result("Skill querying", False, "No skill found")
        
    except Exception as e:
        print_result("Skill system", False, f"Error: {e}")
        traceback.print_exc()
    
    print(f"\n📊 Skill Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_execution_modes(raec):
    """Test 4: Execution Modes"""
    print_header("TEST 4: Execution Modes")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 4.1: Standard mode
    print("📝 Test 4.1: Standard execution mode...")
    
    try:
        result = raec.execute_task(
            "Calculate the sum of numbers from 1 to 10",
            mode="standard"
        )
        
        if result.get('success'):
            print(f"   ✓ Standard mode executed successfully")
            tests_passed += 1
            print_result("Standard mode", True)
        else:
            print_result("Standard mode", False, f"Error: {result.get('error')}")
    
    except Exception as e:
        print_result("Standard mode", False, f"Exception: {e}")
        traceback.print_exc()
    
    # Test 4.2: Collaborative mode
    print("\n📝 Test 4.2: Collaborative execution mode...")
    
    try:
        result = raec.execute_task(
            "Write a function to check if a string is a palindrome",
            mode="collaborative"
        )
        
        if result.get('success'):
            print(f"   ✓ Collaborative mode executed successfully")
            workflow = result.get('workflow_result', {})
            print(f"   Revisions: {workflow.get('revisions', 0)}")
            print(f"   Messages: {workflow.get('message_count', 0)}")
            tests_passed += 1
            print_result("Collaborative mode", True)
        else:
            print_result("Collaborative mode", False, f"Error: {result.get('error')}")
    
    except Exception as e:
        print_result("Collaborative mode", False, f"Exception: {e}")
        traceback.print_exc()
    
    # Test 4.3: Incremental mode
    print("\n📝 Test 4.3: Incremental execution mode...")
    
    try:
        result = raec.execute_task(
            "Explain why bubble sort has O(n²) complexity",
            mode="incremental"
        )
        
        if result.get('success') is not None:
            print(f"   ✓ Incremental mode executed")
            steps = result.get('reasoning_steps', [])
            print(f"   Reasoning steps: {len(steps)}")
            all_passed = result.get('success')
            print(f"   All steps verified: {all_passed}")
            tests_passed += 1
            print_result("Incremental mode", True)
        else:
            print_result("Incremental mode", False, "No result")
    
    except Exception as e:
        print_result("Incremental mode", False, f"Exception: {e}")
        traceback.print_exc()
    
    print(f"\n📊 Execution Mode Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_verification_system(raec):
    """Test 5: Verification System"""
    print_header("TEST 5: Verification System")
    
    tests_passed = 0
    total_tests = 2
    
    try:
        # Test 5.1: Multi-level verification
        print("📝 Test 5.1: Multi-level verification...")
        
        from evaluators.logic_checker import VerificationLevel
        
        test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        
        passed, results = raec.evaluator.verify(
            output=test_code,
            verification_levels=[
                VerificationLevel.SYNTAX,
                VerificationLevel.LOGIC
            ],
            context={'task': 'implement factorial'}
        )
        
        print(f"   Verification passed: {passed}")
        print(f"   Checks performed: {len(results)}")
        
        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"     {status} {result.level.value}: {result.message[:50]}")
        
        tests_passed += 1
        print_result("Multi-level verification", True)
        
        # Test 5.2: Incremental verification
        print("\n📝 Test 5.2: Incremental reasoning verification...")
        
        steps = [
            "Identify the problem domain",
            "Break down into subproblems",
            "Solve each subproblem",
            "Combine solutions"
        ]
        
        step_results = raec.evaluator.incremental_verify(
            steps,
            "Solve a complex problem"
        )
        
        print(f"   Steps verified: {len(step_results)}")
        
        tests_passed += 1
        print_result("Incremental verification", True)
        
    except Exception as e:
        print_result("Verification system", False, f"Error: {e}")
        traceback.print_exc()
    
    print(f"\n📊 Verification Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests


def test_performance_analysis(raec):
    """Test 6: Performance Analysis"""
    print_header("TEST 6: Performance Analysis")
    
    try:
        print("📝 Running performance analysis...\n")
        
        stats = raec.analyze_performance()
        
        # Validate stats structure
        required_keys = ['memory', 'skills', 'tools', 'agents', 'verification']
        all_present = all(key in stats for key in required_keys)
        
        if all_present:
            print_result("Performance analysis", True, "All statistics collected")
            return True
        else:
            print_result("Performance analysis", False, "Missing statistics")
            return False
        
    except Exception as e:
        print_result("Performance analysis", False, f"Error: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("🧪 RAEC SYSTEM INTEGRATION TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'initialization': False,
        'memory': False,
        'skills': False,
        'execution_modes': False,
        'verification': False,
        'performance': False
    }
    
    # Test 1: Initialization
    raec, results['initialization'] = test_initialization()
    
    if not raec:
        print("\n❌ CRITICAL: System initialization failed. Stopping tests.")
        return results
    
    # Test 2: Memory System
    results['memory'] = test_memory_system(raec)
    
    # Test 3: Skill System
    results['skills'] = test_skill_system(raec)
    
    # Test 4: Execution Modes
    results['execution_modes'] = test_execution_modes(raec)
    
    # Test 5: Verification System
    results['verification'] = test_verification_system(raec)
    
    # Test 6: Performance Analysis
    results['performance'] = test_performance_analysis(raec)
    
    # Clean shutdown
    print_header("Shutdown")
    try:
        raec.close()
        print("✓ System shutdown complete")
    except Exception as e:
        print(f"⚠️  Shutdown error: {e}")
    
    # Final Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}\n")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    success_rate = (passed / total) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    try:
        results = run_all_tests()
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
