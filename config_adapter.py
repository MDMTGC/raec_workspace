"""
Configuration Compatibility Helper

Ensures Raec works with both old and new config structures
"""
from typing import Any, Dict, Optional


class ConfigAdapter:
    """
    Adapts between different config file structures
    Provides backward compatibility and sensible defaults
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config
        self._cache = {}
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration with defaults"""
        if 'model' in self.raw_config:
            return self.raw_config['model']
        
        # Fallback for old configs or missing section
        return {
            'name': 'raec:latest',
            'device': 'cuda',
            'temperature': 0.7,
            'max_tokens': 2048
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration with defaults"""
        if 'memory' in self.raw_config:
            return self.raw_config['memory']
        
        return {
            'db_path': 'memory/raec_memory.db',
            'use_cache': True,
            'max_embeddings': 50000
        }
    
    def get_tools_config(self) -> Dict[str, Any]:
        """Get tools configuration with defaults"""
        if 'tools' in self.raw_config:
            return self.raw_config['tools']
        
        return {
            'python_timeout': 60,
            'max_api_fetch_chars': 10000,
            'enable_code_execution': True,
            'enable_file_operations': True,
            'enable_web_access': False
        }
    
    def get_skills_config(self) -> Dict[str, Any]:
        """Get skills configuration with defaults"""
        if 'skills' in self.raw_config:
            return self.raw_config['skills']
        
        return {
            'storage_path': 'skills/skill_db.json',
            'auto_extract': True,
            'verification_threshold': 0.8
        }
    
    def get_planner_config(self) -> Dict[str, Any]:
        """Get planner configuration with defaults"""
        if 'planner' in self.raw_config:
            return self.raw_config['planner']
        
        return {
            'max_steps': 10,
            'default_language': 'python',
            'max_lines_per_script': 1000,
            'auto_format': True
        }
    
    def get_agents_config(self) -> Dict[str, Any]:
        """Get agents configuration with defaults"""
        if 'agents' in self.raw_config:
            return self.raw_config['agents']
        
        # Check legacy settings
        multi_agent = self.raw_config.get('multi_agent_mode', True)
        
        return {
            'enabled': multi_agent,
            'max_messages': 50,
            'max_revisions': 2
        }
    
    def get_evaluator_config(self) -> Dict[str, Any]:
        """Get evaluator configuration with defaults"""
        if 'evaluator' in self.raw_config:
            return self.raw_config['evaluator']
        
        # Check legacy settings
        strict = self.raw_config.get('evaluator_strict_mode', True)
        max_runtime = self.raw_config.get('max_script_runtime', 60)
        
        return {
            'strict_mode': strict,
            'max_script_runtime': max_runtime,
            'verification_levels': ['syntax', 'logic', 'output']
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration with defaults"""
        if 'logging' in self.raw_config:
            return self.raw_config['logging']
        
        # Check legacy settings
        enabled = self.raw_config.get('log_output', True)
        log_path = self.raw_config.get('log_path', 'logs/raec_execution.log')
        
        return {
            'enabled': enabled,
            'log_path': log_path,
            'log_level': 'INFO',
            'log_tasks': True
        }
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration with defaults"""
        if 'gui' in self.raw_config:
            return self.raw_config['gui']
        
        # Check legacy settings
        enabled = self.raw_config.get('gui_enabled', True)
        
        return {
            'enabled': enabled,
            'theme': 'dark',
            'auto_refresh_ms': 2000,
            'max_history': 20
        }
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration with defaults"""
        if 'debug' in self.raw_config:
            return self.raw_config['debug']
        
        # Check legacy settings
        enabled = self.raw_config.get('debug_mode', False)
        
        return {
            'enabled': enabled,
            'verbose_logging': False,
            'save_intermediate_steps': False
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration with defaults"""
        if 'performance' in self.raw_config:
            return self.raw_config['performance']
        
        return {
            'max_concurrent_tasks': 1,
            'memory_limit_mb': 4096,
            'cache_size': 100
        }
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        
        Examples:
            get('model.name') -> 'raec:latest'
            get('tools.python_timeout') -> 60
            get('nonexistent.value', 'default') -> 'default'
        """
        keys = path.split('.')
        value = self.raw_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


def load_config_with_compatibility(config_dict: Dict[str, Any]) -> ConfigAdapter:
    """
    Load config and wrap in compatibility adapter
    
    Args:
        config_dict: Raw config dictionary from YAML
        
    Returns:
        ConfigAdapter instance with backward compatibility
    """
    return ConfigAdapter(config_dict)


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Test with new config structure
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    adapter = load_config_with_compatibility(config_dict)
    
    print("Model Config:", adapter.get_model_config())
    print("Tools Config:", adapter.get_tools_config())
    print("Skills Config:", adapter.get_skills_config())
    print("\nDot notation access:")
    print("model.name:", adapter.get('model.name'))
    print("tools.python_timeout:", adapter.get('tools.python_timeout'))
    print("nonexistent.key:", adapter.get('nonexistent.key', 'DEFAULT'))
