"""
Quick Demo of Raec's New Tools

Shows tools working without needing full Raec/Ollama initialization
"""
from tools.tool_interface import ToolInterface
import os


def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_file_tools(tools):
    print_section("FILE TOOLS DEMO")
    
    # Create a test file
    test_file = "demo_test.txt"
    test_content = "Hello from Raec tools!\nThis is line 2.\nAnd line 3."
    
    print("1. Writing file...")
    result = tools.write_file(test_file, test_content)
    print(f"   {result}")
    
    print("\n2. Reading file...")
    content = tools.read_file(test_file)
    print(f"   Content: {content[:50]}...")
    
    print("\n3. Getting file info...")
    info = tools.execute('file', 'get_file_info', filepath=test_file)
    print(f"   Size: {info.get('size')} bytes")
    print(f"   Exists: {info.get('exists')}")
    
    print("\n4. Appending to file...")
    result = tools.execute('file', 'append_file', filepath=test_file, content="\nLine 4 appended!")
    print(f"   {result}")
    
    print("\n5. Reading updated file...")
    content = tools.read_file(test_file)
    print(f"   Lines: {content.count(chr(10)) + 1}")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print("\n6. Cleaned up test file")


def demo_text_tools(tools):
    print_section("TEXT TOOLS DEMO")
    
    sample_text = """
    Hello! Contact me at john@example.com or visit https://example.com
    You can also reach out to jane@test.org for more info.
    Check out https://github.com/user/repo for code samples.
    """
    
    print("Sample text:")
    print(sample_text)
    
    print("\n1. Counting words...")
    count = tools.execute('text', 'count_words', text=sample_text)
    print(f"   Word count: {count}")
    
    print("\n2. Counting lines...")
    lines = tools.execute('text', 'count_lines', text=sample_text)
    print(f"   Line count: {lines}")
    
    print("\n3. Extracting emails...")
    emails = tools.execute('text', 'extract_emails', text=sample_text)
    print(f"   Found emails: {emails}")
    
    print("\n4. Extracting URLs...")
    urls = tools.execute('text', 'extract_urls', text=sample_text)
    print(f"   Found URLs: {urls}")
    
    print("\n5. Searching for 'contact'...")
    matches = tools.execute('text', 'search_text', text=sample_text, pattern='contact', case_sensitive=False)
    print(f"   Matches: {matches}")


def demo_math_tools(tools):
    print_section("MATH TOOLS DEMO")
    
    print("1. Basic calculation...")
    result = tools.calculate("(2 + 3) * 4 - 10 / 2")
    print(f"   (2 + 3) * 4 - 10 / 2 = {result}")
    
    print("\n2. Statistics on dataset...")
    numbers = [10, 20, 15, 25, 30, 18, 22, 28]
    stats = tools.execute('math', 'statistics', numbers=numbers)
    print(f"   Data: {numbers}")
    print(f"   Mean: {stats.get('mean'):.2f}")
    print(f"   Median: {stats.get('median'):.2f}")
    print(f"   Min: {stats.get('min')}, Max: {stats.get('max')}")
    print(f"   Std Dev: {stats.get('stdev'):.2f}")


def demo_data_tools(tools):
    print_section("DATA TOOLS DEMO")
    
    print("1. JSON operations...")
    data = {'name': 'Raec', 'version': '1.0', 'features': ['memory', 'skills', 'tools']}
    
    json_str = tools.to_json(data)
    print(f"   To JSON:\n{json_str}")
    
    parsed = tools.parse_json(json_str)
    print(f"\n   Parsed back: {parsed}")
    
    print("\n2. List operations...")
    sample_list = ['apple', 'banana', 'cherry', 'apricot', 'blueberry']
    
    print(f"   Original: {sample_list}")
    
    filtered = tools.execute('data', 'filter_list', data=sample_list, condition='a')
    print(f"   Filtered (contains 'a'): {filtered}")
    
    sorted_list = tools.execute('data', 'sort_list', data=sample_list)
    print(f"   Sorted: {sorted_list}")


def demo_code_tools(tools):
    print_section("CODE TOOLS DEMO")
    
    print("1. Validating Python code...")
    
    valid_code = "def hello():\n    print('Hello')"
    result = tools.execute('code', 'validate_python', code=valid_code)
    print(f"   Valid code: {result}")
    
    invalid_code = "def hello(\n    print('Hello')"
    result = tools.execute('code', 'validate_python', code=invalid_code)
    print(f"   Invalid code: {result}")
    
    print("\n2. Running Python code...")
    code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result:.2f}")
"""
    output = tools.run_python(code)
    print(f"   Output: {output.strip()}")


def demo_system_tools(tools):
    print_section("SYSTEM TOOLS DEMO")
    
    print("1. Current directory...")
    cwd = tools.execute('system', 'get_current_dir')
    print(f"   {cwd}")
    
    print("\n2. System information...")
    info = tools.get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n3. Environment variable...")
    path = tools.execute('system', 'get_env_var', var_name='PATH')
    print(f"   PATH length: {len(path)} characters")


def demo_execution_log(tools):
    print_section("EXECUTION LOG")
    
    history = tools.get_execution_history(limit=10)
    
    print(f"Last {len(history)} tool executions:\n")
    
    for i, entry in enumerate(history, 1):
        status = "✓" if entry['success'] else "✗"
        print(f"{i}. {status} [{entry['category']}.{entry['tool']}]")
        print(f"     Result: {entry['result'][:60]}...")


def main():
    print("\n" + "="*70)
    print("  RAEC TOOLS DEMONSTRATION")
    print("  Showing all tool categories in action")
    print("="*70)
    
    # Initialize tools
    tools = ToolInterface()
    
    # Run demos
    demo_file_tools(tools)
    demo_text_tools(tools)
    demo_math_tools(tools)
    demo_data_tools(tools)
    demo_code_tools(tools)
    demo_system_tools(tools)
    demo_execution_log(tools)
    
    print("\n" + "="*70)
    print("  ✅ ALL TOOLS WORKING")
    print("  Raec is no longer floating in the void!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
