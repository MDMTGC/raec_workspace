"""
Static Code Validation for Raec System
Checks all components without requiring Ollama to be running
"""
import os
import sys
import ast
import json
from pathlib import Path


def print_header(title):
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80 + "\n")


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False


def check_python_syntax(filepath):
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_imports(filepath, expected_imports):
    """Check if file contains expected imports"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        imports_found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports_found.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.add(alias.name)
        
        found_count = sum(1 for imp in expected_imports if imp in imports_found)
        return found_count, len(expected_imports), imports_found
        
    except Exception as e:
        return 0, len(expected_imports), set()


def validate_structure():
    """Validate complete Raec directory structure"""
    print_header("FILE STRUCTURE VALIDATION")
    
    base_path = Path("C:/Users/MDMTGC/Desktop/raec_workspace")
    
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_path}")
        return False
    
    print(f"📁 Base directory: {base_path}\n")
    
    structure_valid = True
    
    # Core files
    print("Core Files:")
    core_files = [
        ("config.yaml", "Configuration"),
        ("main.py", "Main integration"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
    ]
    
    for filename, desc in core_files:
        filepath = base_path / filename
        if not check_file_exists(str(filepath), desc):
            structure_valid = False
    
    # Component directories
    print("\nComponent Directories:")
    components = [
        ("memory", "Hierarchical Memory System"),
        ("skills", "Skill Graph"),
        ("tools", "Tool Executor"),
        ("agents", "Multi-Agent Orchestrator"),
        ("evaluators", "Verification System"),
        ("planner", "Planning System"),
        ("raec_core", "Core LLM Interface"),
    ]
    
    for dirname, desc in components:
        dirpath = base_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            print(f"✅ {desc}: {dirname}/")
        else:
            print(f"❌ {desc}: {dirname}/ (NOT FOUND)")
            structure_valid = False
    
    # Data directories
    print("\nData Directories:")
    data_dirs = [
        ("data/embeddings", "Memory databases"),
        ("logs/tasks", "Task logs"),
        ("logs/memory_snapshots", "Memory snapshots"),
    ]
    
    for dirname, desc in data_dirs:
        dirpath = base_path / dirname
        if dirpath.exists():
            print(f"✅ {desc}: {dirname}/")
        else:
            print(f"⚠️  {desc}: {dirname}/ (will be created on first run)")
    
    return structure_valid


def validate_python_files():
    """Validate syntax of all Python files"""
    print_header("PYTHON SYNTAX VALIDATION")
    
    base_path = Path("C:/Users/MDMTGC/Desktop/raec_workspace")
    
    python_files = [
        ("main.py", "Main Integration"),
        ("memory/memory_db.py", "Memory System"),
        ("skills/skill_graph.py", "Skill Graph"),
        ("tools/executor.py", "Tool Executor"),
        ("agents/orchestrator.py", "Multi-Agent System"),
        ("evaluators/logic_checker.py", "Verification System"),
        ("planner/planner.py", "Planner"),
        ("raec_core/llm_interface.py", "LLM Interface"),
    ]
    
    all_valid = True
    
    for filepath, desc in python_files:
        full_path = base_path / filepath
        
        if not full_path.exists():
            print(f"❌ {desc}: {filepath} (NOT FOUND)")
            all_valid = False
            continue
        
        valid, error = check_python_syntax(str(full_path))
        
        if valid:
            size = os.path.getsize(str(full_path))
            lines = sum(1 for _ in open(str(full_path), 'r', encoding='utf-8'))
            print(f"✅ {desc}: {filepath}")
            print(f"   {lines:,} lines, {size:,} bytes")
        else:
            print(f"❌ {desc}: {filepath}")
            print(f"   Syntax Error: {error}")
            all_valid = False
    
    return all_valid


def validate_main_integration():
    """Validate main.py integration"""
    print_header("MAIN.PY INTEGRATION VALIDATION")
    
    base_path = Path("C:/Users/MDMTGC/Desktop/raec_workspace")
    main_path = base_path / "main.py"
    
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    # Check expected imports
    print("Checking imports...")
    expected_imports = [
        "raec_core.llm_interface",
        "planner.planner",
        "memory.memory_db",
        "skills.skill_graph",
        "tools.executor",
        "agents.orchestrator",
        "evaluators.logic_checker"
    ]
    
    found, total, all_imports = check_imports(str(main_path), expected_imports)
    
    print(f"   Expected imports found: {found}/{total}")
    
    if found == total:
        print(f"   ✅ All critical imports present")
    else:
        print(f"   ⚠️  Missing {total - found} import(s)")
        missing = set(expected_imports) - all_imports
        for imp in missing:
            print(f"      - {imp}")
    
    # Check for key classes/functions
    print("\nChecking key components...")
    
    with open(str(main_path), 'r', encoding='utf-8') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find Raec class
    raec_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Raec":
            raec_class = node
            break
    
    if raec_class:
        print("   ✅ Raec class defined")
        
        # Check for key methods
        methods = [m.name for m in raec_class.body if isinstance(m, ast.FunctionDef)]
        
        expected_methods = [
            "__init__",
            "process_input",
            "execute_task",
            "_execute_standard",
            "_execute_collaborative",
            "_execute_incremental",
            "analyze_performance"
        ]
        
        methods_found = [m for m in expected_methods if m in methods]
        
        print(f"   Expected methods: {len(methods_found)}/{len(expected_methods)}")
        
        for method in expected_methods:
            if method in methods:
                print(f"      ✅ {method}")
            else:
                print(f"      ❌ {method}")
        
        return len(methods_found) == len(expected_methods)
    else:
        print("   ❌ Raec class not found")
        return False


def validate_config():
    """Validate configuration file"""
    print_header("CONFIGURATION VALIDATION")
    
    base_path = Path("C:/Users/MDMTGC/Desktop/raec_workspace")
    config_path = base_path / "config.yaml"
    
    if not config_path.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        import yaml
        
        with open(str(config_path), 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ config.yaml is valid YAML")
        
        # Check required sections
        required_sections = ['model', 'memory', 'tools', 'planner', 'skills', 'logs']
        
        print("\nRequired sections:")
        all_present = True
        
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section}")
            else:
                print(f"   ❌ {section}")
                all_present = False
        
        # Check model configuration
        if 'model' in config:
            print("\nModel configuration:")
            print(f"   Model: {config['model'].get('name', 'NOT SET')}")
            print(f"   Device: {config['model'].get('device', 'NOT SET')}")
        
        return all_present
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def count_code_stats():
    """Count code statistics"""
    print_header("CODE STATISTICS")
    
    base_path = Path("C:/Users/MDMTGC/Desktop/raec_workspace")
    
    python_files = list(base_path.glob("**/*.py"))
    
    total_lines = 0
    total_files = 0
    
    component_stats = {}
    
    for filepath in python_files:
        if "__pycache__" in str(filepath):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            
            total_lines += lines
            total_files += 1
            
            # Categorize by component
            relative = filepath.relative_to(base_path)
            component = str(relative.parts[0]) if relative.parts else "root"
            
            if component not in component_stats:
                component_stats[component] = {'files': 0, 'lines': 0}
            
            component_stats[component]['files'] += 1
            component_stats[component]['lines'] += lines
            
        except Exception:
            pass
    
    print(f"Total Python files: {total_files}")
    print(f"Total lines of code: {total_lines:,}\n")
    
    print("Breakdown by component:")
    for component, stats in sorted(component_stats.items()):
        print(f"   {component:20} {stats['files']:3} files  {stats['lines']:6,} lines")
    
    return total_files, total_lines


def main():
    """Run all validation checks"""
    print("\n" + "="*80)
    print("🔍 RAEC STATIC CODE VALIDATION")
    print("="*80)
    print("This validates code structure without requiring Ollama")
    
    results = {}
    
    # Run all checks
    results['structure'] = validate_structure()
    results['syntax'] = validate_python_files()
    results['integration'] = validate_main_integration()
    results['config'] = validate_config()
    
    # Code stats (always succeeds)
    count_code_stats()
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks Passed: {passed}/{total}\n")
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name.replace('_', ' ').title()}")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
        print("✓ Code structure is correct")
        print("✓ All Python files have valid syntax")
        print("✓ Integration is properly configured")
        print("✓ Configuration file is valid")
        print("\nSystem is ready to run (requires Ollama with raec:latest model)")
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("Review errors above and fix before running.")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
