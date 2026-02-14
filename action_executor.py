"""
Action Executor - FULL AUTONOMOUS EXECUTION

Handles ALL common action patterns IMMEDIATELY without LLM overthinking.
This is the core of Raec's autonomous operation.
"""
import re
import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path


class ActionExecutor:
    """
    Aggressive immediate execution engine
    
    Detects action patterns and executes INSTANTLY using tools.
    NO LLM thinking, NO planning delays, JUST ACTION.
    """
    
    def __init__(self, tools, memory=None):
        """
        Initialize with tool interface

        Args:
            tools: ToolInterface instance for actual execution
            memory: Optional HierarchicalMemoryDB for post-execution learning (W8)
        """
        self.tools = tools
        self.memory = memory
        self.execution_log = []
    
    def execute_action(self, task: str) -> Dict[str, Any]:
        """
        Main execution entry point
        
        Detects action type and executes IMMEDIATELY.
        
        Args:
            task: User's task description
            
        Returns:
            Execution result with success status
        """
        start_time = time.time()
        action_type = self.detect_action_type(task)
        
        if not action_type:
            return {
                'success': False,
                'action_type': None,
                'message': 'Cannot detect action type - needs planner'
            }
        
        # Route to appropriate handler
        handlers = {
            'structure_create': self._execute_structure_creation,
            'file_create': self._execute_file_creation,
            'file_read': self._execute_file_read,
            'folder_create': self._execute_folder_creation,
            'code_execute': self._execute_code,
            'math_calculate': self._execute_calculation,
            'web_fetch': self._execute_web_fetch,
            'file_list': self._execute_file_list,
            'text_search': self._execute_text_search,
            'data_process': self._execute_data_process
        }
        
        handler = handlers.get(action_type)
        if not handler:
            return {
                'success': False,
                'action_type': action_type,
                'message': f'No handler for {action_type}'
            }
        
        # EXECUTE
        result = handler(task)
        result['execution_time'] = time.time() - start_time
        result['action_type'] = action_type

        # Log
        self.execution_log.append({
            'task': task,
            'action_type': action_type,
            'success': result.get('success'),
            'time': result['execution_time']
        })

        # W8: Post-execution learning hook — store in memory
        self._store_execution_memory(task, action_type, result)

        return result
    
    def detect_action_type(self, task: str) -> Optional[str]:
        """
        Detect what action the user wants
        
        Returns action type or None
        """
        task_lower = task.lower()
        
        # File/folder structure operations
        if any(word in task_lower for word in [
            'file structure', 'folder structure', 'directory structure',
            'myeverydaymaterials', 'create structure'
        ]):
            return 'structure_create'
        
        # Individual file operations
        if any(word in task_lower for word in ['create file', 'make file', 'write file']):
            return 'file_create'
        
        if any(word in task_lower for word in ['read file', 'open file', 'show file', 'cat ']):
            return 'file_read'
        
        if any(word in task_lower for word in ['list files', 'list directory', 'ls ', 'dir ']):
            return 'file_list'
        
        # Folder operations
        if any(word in task_lower for word in [
            'create folder', 'make folder', 'create directory',
            'mkdir', 'make dir'
        ]):
            return 'folder_create'
        
        # Code execution
        if any(word in task_lower for word in [
            'run script', 'execute script', 'run python', 'execute python',
            'run code', 'execute code', 'run this', 'execute this'
        ]):
            return 'code_execute'
        
        # Math operations
        if any(word in task_lower for word in [
            'calculate', 'compute', 'solve', 'what is',
            'how much', 'evaluate'
        ]) and any(op in task_lower for op in ['+', '-', '*', '/', '^', '=']):
            return 'math_calculate'
        
        # Web operations
        if any(word in task_lower for word in [
            'download', 'fetch', 'get url', 'http', 'https',
            'web page', 'website'
        ]):
            return 'web_fetch'
        
        # Text operations
        if any(word in task_lower for word in [
            'search for', 'find in', 'grep', 'search text'
        ]):
            return 'text_search'
        
        # Data processing
        if any(word in task_lower for word in [
            'parse json', 'parse csv', 'convert', 'transform data'
        ]):
            return 'data_process'
        
        return None
    
    def _execute_structure_creation(self, task: str) -> Dict[str, Any]:
        """Create file/folder structure"""
        
        # MyEverydayMaterials specific pattern
        if 'myeverydaymaterials' in task.lower():
            return self._create_materials_structure()
        
        # Generic structure creation would need more parsing
        return {
            'success': False,
            'message': 'Generic structure creation needs more implementation'
        }
    
    def _create_materials_structure(self) -> Dict[str, Any]:
        """
        Create MyEverydayMaterials website structure
        
        This is a REAL, COMPLETE implementation that ACTUALLY DOES IT.
        """
        base_path = r"C:\Users\MDMTGC\Desktop\MyEverydayMaterials"
        
        structure = {
            'root': ['index.html'],
            'css': ['style.css'],
            'metals': [
                'aluminum-rust.html',
                'magnets.html',
                'stainless-steel.html',
                'microwaving-metal.html',
                'copper-patina.html'
            ]
        }
        
        created_items = []
        
        try:
            # Create root folder
            os.makedirs(base_path, exist_ok=True)
            created_items.append(f"📁 {base_path}")
            
            # Create root files
            for filename in structure['root']:
                filepath = os.path.join(base_path, filename)
                result = self.tools.write_file(filepath, '')
                created_items.append(f"📄 {filename}")
            
            # Create subfolders and their files
            for folder, files in structure.items():
                if folder == 'root':
                    continue
                
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                created_items.append(f"📁 {folder}/")
                
                for filename in files:
                    filepath = os.path.join(folder_path, filename)
                    result = self.tools.write_file(filepath, '')
                    created_items.append(f"📄 {folder}/{filename}")
            
            return {
                'success': True,
                'message': f"Created MyEverydayMaterials structure with {len(created_items)} items",
                'details': created_items,
                'path': base_path,
                'files_created': len([x for x in created_items if '📄' in x]),
                'folders_created': len([x for x in created_items if '📁' in x])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to create structure: {e}",
                'partial_results': created_items
            }
    
    def _execute_file_creation(self, task: str) -> Dict[str, Any]:
        """Create a single file"""
        
        # Extract filename - simplified pattern matching
        # TODO: More sophisticated filename extraction
        
        match = re.search(r'create file ["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        if not match:
            match = re.search(r'file called ["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        
        if not match:
            return {
                'success': False,
                'message': 'Could not extract filename from task'
            }
        
        filename = match.group(1).strip()
        
        try:
            result = self.tools.write_file(filename, '')
            return {
                'success': True,
                'message': f"Created file: {filename}",
                'filename': filename
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to create {filename}: {e}"
            }
    
    def _execute_file_read(self, task: str) -> Dict[str, Any]:
        """Read a file"""
        
        # Extract filename
        match = re.search(r'(?:read|open|show|cat)\s+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        
        if not match:
            return {
                'success': False,
                'message': 'Could not extract filename from task'
            }
        
        filename = match.group(1).strip()
        
        try:
            content = self.tools.read_file(filename)
            return {
                'success': True,
                'message': f"Read {filename} ({len(content)} bytes)",
                'filename': filename,
                'content': content,
                'size': len(content)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to read {filename}: {e}"
            }
    
    def _execute_folder_creation(self, task: str) -> Dict[str, Any]:
        """Create folders"""
        
        # Extract folder name
        match = re.search(r'(?:create|make|mkdir)\s+(?:folder|directory)\s+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        
        if not match:
            return {
                'success': False,
                'message': 'Could not extract folder name from task'
            }
        
        folder_name = match.group(1).strip()
        
        try:
            os.makedirs(folder_name, exist_ok=True)
            return {
                'success': True,
                'message': f"Created folder: {folder_name}",
                'folder': folder_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to create {folder_name}: {e}"
            }
    
    def _execute_file_list(self, task: str) -> Dict[str, Any]:
        """List files in directory"""
        
        # Extract directory path
        match = re.search(r'(?:list|ls|dir)\s+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        
        if match:
            path = match.group(1).strip()
        else:
            path = '.'  # Current directory
        
        try:
            listing = self.tools.list_dir(path)
            return {
                'success': True,
                'message': f"Listed contents of {path}",
                'path': path,
                'listing': listing
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to list {path}: {e}"
            }
    
    def _execute_code(self, task: str) -> Dict[str, Any]:
        """Execute Python code"""
        
        # Extract code block
        code_match = re.search(r'```python\n(.*?)\n```', task, re.DOTALL | re.IGNORECASE)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', task, re.DOTALL)
        
        if not code_match:
            # Try to find inline code
            code_match = re.search(r'run\s+(?:this|code|script)[\s:]+(.+)', task, re.IGNORECASE | re.DOTALL)
        
        if not code_match:
            return {
                'success': False,
                'message': 'Could not extract code from task'
            }
        
        code = code_match.group(1).strip()
        
        try:
            output = self.tools.run_python(code)
            return {
                'success': True,
                'message': f"Executed Python code",
                'code': code,
                'output': output
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Code execution failed: {e}",
                'code': code
            }
    
    def _execute_calculation(self, task: str) -> Dict[str, Any]:
        """Perform math calculation"""
        
        # Extract expression
        match = re.search(r'(?:calculate|compute|solve|what is)\s+(.+?)(?:\?|$)', task, re.IGNORECASE)
        
        if not match:
            return {
                'success': False,
                'message': 'Could not extract math expression'
            }
        
        expression = match.group(1).strip()
        
        # Clean up common words
        expression = re.sub(r'\b(equals?|is)\b', '', expression, flags=re.IGNORECASE).strip()
        
        try:
            result = self.tools.calculate(expression)
            return {
                'success': True,
                'message': f"{expression} = {result}",
                'expression': expression,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Calculation failed: {e}",
                'expression': expression
            }
    
    def _execute_web_fetch(self, task: str) -> Dict[str, Any]:
        """Fetch web content"""
        
        # Extract URL
        url_match = re.search(r'(https?://[^\s]+)', task, re.IGNORECASE)
        
        if not url_match:
            return {
                'success': False,
                'message': 'Could not extract URL from task'
            }
        
        url = url_match.group(1).strip()
        
        try:
            content = self.tools.http_get(url)
            return {
                'success': True,
                'message': f"Fetched {url} ({len(content)} bytes)",
                'url': url,
                'content': content[:1000],  # Truncate for display
                'full_size': len(content)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to fetch {url}: {e}",
                'url': url
            }
    
    def _execute_text_search(self, task: str) -> Dict[str, Any]:
        """Search text"""
        
        # Extract search pattern
        match = re.search(r'(?:search for|find)\s+["\']?([^"\']+)["\']?\s+in\s+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
        
        if not match:
            return {
                'success': False,
                'message': 'Could not extract search pattern and file'
            }
        
        pattern = match.group(1).strip()
        filename = match.group(2).strip()
        
        try:
            content = self.tools.read_file(filename)
            matches = self.tools.search_text(content, pattern)
            return {
                'success': True,
                'message': f"Found {len(matches)} matches for '{pattern}' in {filename}",
                'pattern': pattern,
                'filename': filename,
                'matches': matches
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Search failed: {e}"
            }
    
    def _execute_data_process(self, task: str) -> Dict[str, Any]:
        """Process data (JSON, CSV, etc.)"""
        
        # This needs more sophisticated parsing
        # For now, basic JSON parsing
        
        if 'json' in task.lower():
            # Extract JSON string
            json_match = re.search(r'\{.*\}', task, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = self.tools.parse_json(json_str)
                    return {
                        'success': True,
                        'message': 'Parsed JSON',
                        'data': parsed
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'message': f"JSON parsing failed: {e}"
                    }
        
        return {
            'success': False,
            'message': 'Data processing needs more specific implementation'
        }
    
    def _store_execution_memory(self, task: str, action_type: str, result: Dict[str, Any]):
        """
        W8: Post-execution learning hook.
        Stores fast-path execution results as EXPERIENCE memories so the
        system can learn from ActionExecutor runs, not just planner runs.
        Only stores failures and notable successes to avoid memory bloat.
        """
        if not self.memory:
            return

        try:
            from memory.memory_db import MemoryType

            success = result.get('success', False)

            # Always store failures (learning opportunities)
            if not success:
                error = result.get('error', result.get('message', 'unknown'))
                content = (
                    f"ActionExecutor failed: {action_type} for '{task[:100]}' — {str(error)[:200]}"
                )
                self.memory.store(
                    content=content,
                    memory_type=MemoryType.EXPERIENCE,
                    metadata={
                        'action_type': action_type,
                        'task': task[:200],
                        'success': False,
                        'error': str(error)[:300],
                        'execution_time': result.get('execution_time'),
                    },
                    confidence=0.5,
                    source='action_executor'
                )
            else:
                # Store successes only periodically (every 5th) to avoid bloat
                success_count = sum(
                    1 for e in self.execution_log if e.get('success')
                )
                if success_count % 5 == 0:
                    content = (
                        f"ActionExecutor succeeded: {action_type} for '{task[:100]}' "
                        f"in {result.get('execution_time', 0):.2f}s"
                    )
                    self.memory.store(
                        content=content,
                        memory_type=MemoryType.EXPERIENCE,
                        metadata={
                            'action_type': action_type,
                            'task': task[:200],
                            'success': True,
                            'execution_time': result.get('execution_time'),
                        },
                        confidence=1.0,
                        source='action_executor'
                    )
        except Exception as e:
            # Non-critical — don't fail the action for a memory error
            pass

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e['success'])
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'recent': self.execution_log[-10:]
        }


# Example usage
if __name__ == "__main__":
    from tools.tool_interface import ToolInterface
    
    tools = ToolInterface()
    executor = ActionExecutor(tools)
    
    # Test various action types
    test_tasks = [
        "Create the MyEverydayMaterials file structure",
        "Calculate 2 + 2 * 3",
        "Create file test.txt",
        "List files in current directory",
        "Run this Python code: print('Hello')"
    ]
    
    print("ACTION EXECUTOR TEST\n" + "="*70)
    
    for task in test_tasks:
        print(f"\nTask: {task}")
        action_type = executor.detect_action_type(task)
        print(f"Detected: {action_type}")
        
        if action_type:
            result = executor.execute_action(task)
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
    
    print("\n" + "="*70)
    print("STATS:", executor.get_execution_stats())
