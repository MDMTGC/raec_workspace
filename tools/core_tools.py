"""
Core Tool Library for Raec

Provides essential tools for:
- File operations
- Web requests
- Data processing
- Code execution
- System operations
"""
import os
import json
import subprocess
import tempfile
from typing import Any, Dict, Optional, List
from pathlib import Path


class FileTools:
    """File system operations"""
    
    @staticmethod
    def read_file(filepath: str) -> str:
        """Read a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    def write_file(filepath: str, content: str) -> str:
        """Write content to a file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {filepath}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    def append_file(filepath: str, content: str) -> str:
        """Append content to a file"""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully appended to {filepath}"
        except Exception as e:
            return f"Error appending to file: {e}"
    
    @staticmethod
    def list_directory(dirpath: str) -> str:
        """List files in a directory"""
        try:
            items = os.listdir(dirpath)
            result = []
            for item in items:
                full_path = os.path.join(dirpath, item)
                if os.path.isdir(full_path):
                    result.append(f"[DIR]  {item}")
                else:
                    size = os.path.getsize(full_path)
                    result.append(f"[FILE] {item} ({size:,} bytes)")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """Check if file exists"""
        return os.path.exists(filepath)
    
    @staticmethod
    def delete_file(filepath: str) -> str:
        """Delete a file"""
        try:
            os.remove(filepath)
            return f"Deleted {filepath}"
        except Exception as e:
            return f"Error deleting file: {e}"
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            stat = os.stat(filepath)
            return {
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_file': os.path.isfile(filepath),
                'is_dir': os.path.isdir(filepath),
                'exists': True
            }
        except Exception as e:
            return {'exists': False, 'error': str(e)}


class WebTools:
    """Web and API operations"""
    
    @staticmethod
    def http_get(url: str, timeout: int = 10) -> str:
        """Make HTTP GET request"""
        try:
            import requests
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"HTTP GET error: {e}"
    
    @staticmethod
    def http_post(url: str, data: Dict, timeout: int = 10) -> str:
        """Make HTTP POST request"""
        try:
            import requests
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            return response.text
        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"HTTP POST error: {e}"
    
    @staticmethod
    def download_file(url: str, save_path: str) -> str:
        """Download file from URL"""
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return f"Downloaded {len(response.content)} bytes to {save_path}"
        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"Download error: {e}"


class DataTools:
    """Data processing operations"""
    
    @staticmethod
    def parse_json(json_str: str) -> Any:
        """Parse JSON string"""
        try:
            return json.loads(json_str)
        except Exception as e:
            return {'error': f"JSON parse error: {e}"}
    
    @staticmethod
    def to_json(data: Any, pretty: bool = True) -> str:
        """Convert data to JSON string"""
        try:
            if pretty:
                return json.dumps(data, indent=2)
            return json.dumps(data)
        except Exception as e:
            return f"JSON conversion error: {e}"
    
    @staticmethod
    def parse_csv(csv_str: str, delimiter: str = ',') -> List[List[str]]:
        """Parse CSV string"""
        try:
            import csv
            import io
            reader = csv.reader(io.StringIO(csv_str), delimiter=delimiter)
            return list(reader)
        except Exception as e:
            return [['error', str(e)]]
    
    @staticmethod
    def to_csv(data: List[List[Any]], delimiter: str = ',') -> str:
        """Convert data to CSV string"""
        try:
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output, delimiter=delimiter)
            writer.writerows(data)
            return output.getvalue()
        except Exception as e:
            return f"CSV conversion error: {e}"
    
    @staticmethod
    def filter_list(data: List[Any], condition: str) -> List[Any]:
        """Filter list based on condition (string contains or extension patterns)"""
        try:
            # If data is a string (e.g., newline-separated), split it
            if isinstance(data, str):
                # Handle both real newlines and escaped newlines (\\n)
                import re
                data = re.split(r'\\n|\n', data)
                data = [line.strip() for line in data if line.strip()]

            condition_lower = condition.lower().strip()

            # Extract extension pattern from various formats
            # Handles: ".py", "*.py", "ends with .py", "extension == '.py'", etc.
            import re
            ext_match = re.search(r'[\'"]?(\.\w+)[\'"]?', condition)
            if ext_match:
                suffix = ext_match.group(1).lower()
                # Check if suffix appears in the item
                return [item for item in data if suffix in str(item).lower()]

            # Handle "ends with X" or "endswith X" patterns without extension
            if 'ends' in condition_lower and 'with' in condition_lower:
                parts = condition_lower.split()
                suffix = parts[-1].strip('\'"')
                return [item for item in data if suffix in str(item).lower()]

            # Default: simple contains
            return [item for item in data if condition_lower in str(item).lower()]
        except Exception as e:
            return [f"Filter error: {e}"]
    
    @staticmethod
    def count(data: Any) -> int:
        """Count items in a list or characters/lines in a string"""
        try:
            if isinstance(data, str):
                # Handle escaped newlines
                import re
                lines = re.split(r'\\n|\n', data)
                lines = [line.strip() for line in lines if line.strip()]
                return len(lines)
            elif isinstance(data, (list, tuple, set)):
                return len(data)
            elif isinstance(data, dict):
                return len(data)
            else:
                return 1
        except Exception as e:
            return f"Count error: {e}"

    @staticmethod
    def sort_list(data: List[Any], reverse: bool = False) -> List[Any]:
        """Sort a list"""
        try:
            return sorted(data, reverse=reverse)
        except Exception as e:
            return [f"Sort error: {e}"]


class CodeTools:
    """Code execution and analysis"""
    
    @staticmethod
    def run_python(code: str, timeout: int = 10, cwd: str = None) -> str:
        """Execute Python code safely, optionally in a specified working directory"""
        try:
            # Use current working directory if not specified
            work_dir = cwd or os.getcwd()

            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )

            os.remove(temp_path)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error:\n{result.stderr}"
        
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out"
        except Exception as e:
            return f"Execution error: {e}"
    
    @staticmethod
    def run_shell(command: str, timeout: int = 10) -> str:
        """Execute shell command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error:\n{result.stderr}"
        
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Command error: {e}"
    
    @staticmethod
    def validate_python(code: str) -> Dict[str, Any]:
        """Check if Python code is syntactically valid"""
        import ast
        try:
            ast.parse(code)
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset
            }


class TextTools:
    """Text processing operations"""
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    @staticmethod
    def count_lines(text: str) -> int:
        """Count lines in text"""
        return len(text.split('\n'))
    
    @staticmethod
    def search_text(text: str, pattern: str, case_sensitive: bool = False) -> List[str]:
        """Search for pattern in text"""
        import re
        flags = 0 if case_sensitive else re.IGNORECASE
        matches = re.findall(pattern, text, flags)
        return matches
    
    @staticmethod
    def replace_text(text: str, old: str, new: str) -> str:
        """Replace text"""
        return text.replace(old, new)
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        import re
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        import re
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)


class MathTools:
    """Mathematical operations"""
    
    @staticmethod
    def calculate(expression: str) -> Any:
        """Safely evaluate mathematical expression"""
        try:
            # Only allow safe operations
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return result
        except Exception as e:
            return f"Calculation error: {e}"
    
    @staticmethod
    def statistics(numbers: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of numbers"""
        try:
            import statistics as stats
            return {
                'mean': stats.mean(numbers),
                'median': stats.median(numbers),
                'stdev': stats.stdev(numbers) if len(numbers) > 1 else 0,
                'min': min(numbers),
                'max': max(numbers),
                'sum': sum(numbers),
                'count': len(numbers)
            }
        except Exception as e:
            return {'error': str(e)}


class SystemTools:
    """System information and operations"""
    
    @staticmethod
    def get_env_var(var_name: str) -> str:
        """Get environment variable"""
        return os.environ.get(var_name, f"Environment variable '{var_name}' not found")
    
    @staticmethod
    def set_env_var(var_name: str, value: str) -> str:
        """Set environment variable"""
        try:
            os.environ[var_name] = value
            return f"Set {var_name}={value}"
        except Exception as e:
            return f"Error setting environment variable: {e}"
    
    @staticmethod
    def get_current_dir() -> str:
        """Get current working directory"""
        return os.getcwd()
    
    @staticmethod
    def change_dir(path: str) -> str:
        """Change working directory"""
        try:
            os.chdir(path)
            return f"Changed to {os.getcwd()}"
        except Exception as e:
            return f"Error changing directory: {e}"
    
    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """Get system information"""
        import platform
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }


# Tool registry for easy access
TOOL_REGISTRY = {
    'file': FileTools,
    'web': WebTools,
    'data': DataTools,
    'code': CodeTools,
    'text': TextTools,
    'math': MathTools,
    'system': SystemTools
}


def get_tool(category: str, tool_name: str):
    """Get a tool function by category and name"""
    if category not in TOOL_REGISTRY:
        return None
    
    tool_class = TOOL_REGISTRY[category]
    return getattr(tool_class, tool_name, None)


def list_all_tools() -> Dict[str, List[str]]:
    """List all available tools"""
    result = {}
    for category, tool_class in TOOL_REGISTRY.items():
        methods = [m for m in dir(tool_class) if not m.startswith('_')]
        result[category] = methods
    return result


if __name__ == "__main__":
    # Demo
    print("Available Tools:")
    for category, tools in list_all_tools().items():
        print(f"\n{category.upper()}:")
        for tool in tools:
            print(f"  - {tool}")
