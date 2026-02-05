"""
Advanced Logic Checker and Error Correction
Based on ToolReflection and incremental inference patterns

Provides:
- Multi-level verification (syntax, logic, output)
- Error correction suggestions
- Incremental reasoning verification
- Predicate logic decomposition
"""
import ast
import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class VerificationLevel(Enum):
    """Levels of verification"""
    SYNTAX = "syntax"           # Code syntax checking
    LOGIC = "logic"             # Logical correctness
    OUTPUT = "output"           # Output validation
    SEMANTIC = "semantic"       # Semantic consistency
    PERFORMANCE = "performance" # Performance metrics


class SeverityLevel(Enum):
    """Severity of issues"""
    CRITICAL = "critical"       # Blocks execution
    ERROR = "error"             # Significant problem
    WARNING = "warning"         # Potential issue
    INFO = "info"               # Informational


@dataclass
class VerificationResult:
    """Result of a verification check"""
    level: VerificationLevel
    passed: bool
    severity: SeverityLevel
    message: str
    details: Dict[str, Any]
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'passed': self.passed,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'suggestions': self.suggestions
        }


class LogicChecker:
    """
    Advanced verification and error correction system
    """
    
    def __init__(self, llm_interface=None):
        self.llm = llm_interface
        self.verification_history: List[Dict] = []
    
    def verify(
        self,
        output: Any,
        expected: Optional[Any] = None,
        verification_levels: Optional[List[VerificationLevel]] = None,
        context: Optional[Dict] = None
    ) -> Tuple[bool, List[VerificationResult]]:
        """
        Comprehensive verification of output
        
        Args:
            output: Output to verify
            expected: Expected output (if known)
            verification_levels: Which checks to run
            context: Additional context for verification
            
        Returns:
            (overall_passed, list of verification results)
        """
        if verification_levels is None:
            verification_levels = [
                VerificationLevel.SYNTAX,
                VerificationLevel.LOGIC,
                VerificationLevel.OUTPUT
            ]
        
        results = []
        
        # Run each verification level
        for level in verification_levels:
            if level == VerificationLevel.SYNTAX:
                result = self._verify_syntax(output, context)
            elif level == VerificationLevel.LOGIC:
                result = self._verify_logic(output, context)
            elif level == VerificationLevel.OUTPUT:
                result = self._verify_output(output, expected, context)
            elif level == VerificationLevel.SEMANTIC:
                result = self._verify_semantic(output, context)
            elif level == VerificationLevel.PERFORMANCE:
                result = self._verify_performance(output, context)
            else:
                continue
            
            results.append(result)
        
        # Overall pass: all critical and error-level checks must pass
        overall_passed = all(
            r.passed or r.severity in [SeverityLevel.WARNING, SeverityLevel.INFO]
            for r in results
        )
        
        # Record verification
        self.verification_history.append({
            'output': str(output)[:200],
            'passed': overall_passed,
            'num_checks': len(results),
            'timestamp': __import__('time').time()
        })
        
        return overall_passed, results
    
    def _verify_syntax(self, output: Any, context: Optional[Dict]) -> VerificationResult:
        """Verify syntax correctness (for code)"""
        output_str = str(output)
        
        # Check if output looks like code
        if not any(keyword in output_str for keyword in ['def ', 'class ', 'import ', 'return ', '=']):
            return VerificationResult(
                level=VerificationLevel.SYNTAX,
                passed=True,
                severity=SeverityLevel.INFO,
                message="Output does not appear to be code",
                details={},
                suggestions=[]
            )
        
        try:
            # Try to parse as Python
            ast.parse(output_str)
            return VerificationResult(
                level=VerificationLevel.SYNTAX,
                passed=True,
                severity=SeverityLevel.INFO,
                message="Syntax is valid",
                details={'language': 'python'},
                suggestions=[]
            )
        except SyntaxError as e:
            return VerificationResult(
                level=VerificationLevel.SYNTAX,
                passed=False,
                severity=SeverityLevel.CRITICAL,
                message=f"Syntax error: {e}",
                details={'error': str(e), 'line': e.lineno if hasattr(e, 'lineno') else None},
                suggestions=[
                    "Check for missing colons, parentheses, or brackets",
                    "Verify indentation is correct",
                    "Look for unclosed strings or comments"
                ]
            )
        except Exception as e:
            return VerificationResult(
                level=VerificationLevel.SYNTAX,
                passed=False,
                severity=SeverityLevel.ERROR,
                message=f"Parse error: {e}",
                details={'error': str(e)},
                suggestions=["Review code structure"]
            )
    
    def _verify_logic(self, output: Any, context: Optional[Dict]) -> VerificationResult:
        """Verify logical correctness"""
        issues = []
        suggestions = []
        
        output_str = str(output)
        
        # Check for common logical issues
        
        # 1. Contradiction detection
        contradictions = self._detect_contradictions(output_str)
        if contradictions:
            issues.extend(contradictions)
            suggestions.append("Resolve logical contradictions")
        
        # 2. Incomplete reasoning chains
        if context and 'task' in context:
            task = context['task']
            if not self._addresses_task(output_str, task):
                issues.append("Output may not fully address the task")
                suggestions.append("Ensure all aspects of the task are covered")
        
        # 3. Missing error handling (for code)
        if 'def ' in output_str and 'try:' not in output_str and 'except' not in output_str:
            issues.append("Code lacks error handling")
            suggestions.append("Consider adding try-except blocks for robustness")
        
        if issues:
            return VerificationResult(
                level=VerificationLevel.LOGIC,
                passed=False,
                severity=SeverityLevel.WARNING,
                message=f"Logical issues detected: {'; '.join(issues)}",
                details={'issues': issues},
                suggestions=suggestions
            )
        
        return VerificationResult(
            level=VerificationLevel.LOGIC,
            passed=True,
            severity=SeverityLevel.INFO,
            message="Logic appears sound",
            details={},
            suggestions=[]
        )
    
    def _verify_output(
        self,
        output: Any,
        expected: Optional[Any],
        context: Optional[Dict]
    ) -> VerificationResult:
        """Verify output correctness"""
        
        # If we have expected output, compare directly
        if expected is not None:
            if output == expected:
                return VerificationResult(
                    level=VerificationLevel.OUTPUT,
                    passed=True,
                    severity=SeverityLevel.INFO,
                    message="Output matches expected",
                    details={'expected': expected, 'actual': output},
                    suggestions=[]
                )
            else:
                return VerificationResult(
                    level=VerificationLevel.OUTPUT,
                    passed=False,
                    severity=SeverityLevel.ERROR,
                    message="Output does not match expected",
                    details={'expected': expected, 'actual': output},
                    suggestions=[
                        "Review the logic that produces this output",
                        "Check if expected output is correct",
                        "Consider edge cases that might cause differences"
                    ]
                )
        
        # Check for error indicators
        output_str = str(output).lower()
        error_indicators = ['error', 'failed', 'exception', 'traceback']
        
        if any(indicator in output_str for indicator in error_indicators):
            return VerificationResult(
                level=VerificationLevel.OUTPUT,
                passed=False,
                severity=SeverityLevel.ERROR,
                message="Output contains error indicators",
                details={'output_preview': str(output)[:200]},
                suggestions=["Review error message and fix underlying issue"]
            )
        
        # Check for empty/null output
        if not output or output == "" or output == "None":
            return VerificationResult(
                level=VerificationLevel.OUTPUT,
                passed=False,
                severity=SeverityLevel.WARNING,
                message="Output is empty or null",
                details={},
                suggestions=["Verify that the operation produced results"]
            )
        
        # If we reach here, output seems reasonable
        return VerificationResult(
            level=VerificationLevel.OUTPUT,
            passed=True,
            severity=SeverityLevel.INFO,
            message="Output appears valid",
            details={},
            suggestions=[]
        )
    
    def _verify_semantic(self, output: Any, context: Optional[Dict]) -> VerificationResult:
        """Verify semantic consistency using LLM"""
        if not self.llm:
            return VerificationResult(
                level=VerificationLevel.SEMANTIC,
                passed=True,
                severity=SeverityLevel.INFO,
                message="Semantic verification skipped (no LLM)",
                details={},
                suggestions=[]
            )
        
        # Use LLM to check semantic consistency
        try:
            prompt = f"""
Analyze the following output for semantic consistency and correctness:

Output: {output}

Context: {json.dumps(context or {}, indent=2)}

Check for:
1. Internal consistency
2. Logical flow
3. Completeness
4. Clarity

Respond with: [PASS] or [FAIL] followed by brief explanation.
"""
            
            result = self.llm.generate(prompt, temperature=0.2, max_tokens=200)
            
            passed = result.strip().startswith('[PASS]')
            explanation = result.replace('[PASS]', '').replace('[FAIL]', '').strip()
            
            return VerificationResult(
                level=VerificationLevel.SEMANTIC,
                passed=passed,
                severity=SeverityLevel.WARNING if not passed else SeverityLevel.INFO,
                message=explanation[:200],
                details={'llm_analysis': result},
                suggestions=["Address semantic issues noted by LLM"] if not passed else []
            )
            
        except Exception as e:
            return VerificationResult(
                level=VerificationLevel.SEMANTIC,
                passed=True,
                severity=SeverityLevel.INFO,
                message=f"Semantic verification error: {e}",
                details={},
                suggestions=[]
            )
    
    def _verify_performance(self, output: Any, context: Optional[Dict]) -> VerificationResult:
        """Verify performance characteristics"""
        issues = []
        suggestions = []
        
        # Check for performance anti-patterns in code
        output_str = str(output)
        
        # Nested loops (O(n²) or worse)
        nested_loops = len(re.findall(r'for\s+\w+\s+in\s+\w+:.*?for\s+\w+\s+in', output_str, re.DOTALL))
        if nested_loops > 0:
            issues.append(f"Found {nested_loops} nested loop(s) - potential O(n²) complexity")
            suggestions.append("Consider using more efficient algorithms or data structures")
        
        # Repeated operations in loops
        if 'for ' in output_str and ('append' in output_str or '+=' in output_str):
            if '.append(' in output_str or 'list +=' in output_str:
                issues.append("List concatenation in loop - inefficient")
                suggestions.append("Use list comprehension or pre-allocate list")
        
        # Inefficient string concatenation
        string_concat_in_loop = re.findall(r'for.*?(\w+)\s*\+=\s*["\']', output_str)
        if string_concat_in_loop:
            issues.append("String concatenation in loop")
            suggestions.append("Use ''.join() or list accumulation for better performance")
        
        if issues:
            return VerificationResult(
                level=VerificationLevel.PERFORMANCE,
                passed=True,  # Not blocking, just informational
                severity=SeverityLevel.WARNING,
                message=f"Performance concerns: {'; '.join(issues)}",
                details={'issues': issues},
                suggestions=suggestions
            )
        
        return VerificationResult(
            level=VerificationLevel.PERFORMANCE,
            passed=True,
            severity=SeverityLevel.INFO,
            message="No obvious performance issues",
            details={},
            suggestions=[]
        )
    
    def suggest_correction(
        self,
        output: Any,
        verification_results: List[VerificationResult],
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Generate correction suggestions based on verification results
        
        Args:
            output: Original output that failed verification
            verification_results: Failed verifications
            context: Additional context
            
        Returns:
            Suggested correction or None
        """
        if not self.llm:
            # Return basic suggestions without LLM
            all_suggestions = []
            for result in verification_results:
                if not result.passed:
                    all_suggestions.extend(result.suggestions)
            
            if all_suggestions:
                return "Suggestions:\n" + "\n".join(f"- {s}" for s in all_suggestions)
            return None
        
        # Use LLM to generate detailed correction
        failed_checks = [r for r in verification_results if not r.passed]
        
        if not failed_checks:
            return None
        
        prompt = f"""
The following output has verification issues:

Output:
{output}

Failed Checks:
{json.dumps([r.to_dict() for r in failed_checks], indent=2)}

Context:
{json.dumps(context or {}, indent=2)}

Provide a corrected version that addresses all issues. Explain your corrections.
"""
        
        try:
            correction = self.llm.generate(prompt, temperature=0.3, max_tokens=1024)
            return correction
        except Exception as e:
            return f"Failed to generate correction: {e}"
    
    def incremental_verify(
        self,
        reasoning_steps: List[str],
        task: str
    ) -> List[Tuple[int, bool, str]]:
        """
        Verify each step of incremental reasoning
        Based on incremental inference patterns
        
        Args:
            reasoning_steps: List of reasoning steps
            task: Original task
            
        Returns:
            List of (step_num, passed, feedback) tuples
        """
        results = []
        
        print("\n[?] INCREMENTAL REASONING VERIFICATION")
        print(f"{'='*70}\n")
        
        accumulated_context = f"Task: {task}\n\n"
        
        for i, step in enumerate(reasoning_steps, 1):
            print(f"Step {i}: {step[:60]}...")
            
            accumulated_context += f"Step {i}: {step}\n"
            
            if self.llm:
                # Use LLM to verify step
                prompt = f"""
Verify the following reasoning step:

{accumulated_context}

Is Step {i} logically sound and consistent with the task and previous steps?

Respond: [VALID] or [INVALID] followed by brief explanation.
"""
                try:
                    verification = self.llm.generate(prompt, temperature=0.2, max_tokens=150)
                    passed = verification.strip().startswith('[VALID]')
                    feedback = verification.replace('[VALID]', '').replace('[INVALID]', '').strip()
                    
                    results.append((i, passed, feedback))
                    print(f"  {'[OK]' if passed else '[X]'} {feedback[:50]}...")
                    
                except Exception as e:
                    results.append((i, True, f"Verification error: {e}"))
                    print(f"  ? Verification error")
            else:
                # Simple heuristic verification without LLM
                # Check for contradiction words
                contradictions = ['however', 'but', 'contradicts', 'opposite']
                has_contradiction = any(word in step.lower() for word in contradictions)
                
                passed = not has_contradiction
                feedback = "Contains potential contradiction" if has_contradiction else "Appears consistent"
                
                results.append((i, passed, feedback))
                print(f"  {'[OK]' if passed else '[X]'} {feedback}")
        
        print()
        return results
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect logical contradictions in text"""
        contradictions = []
        
        # Simple pattern-based contradiction detection
        text_lower = text.lower()
        
        # Look for direct contradictions
        contradiction_patterns = [
            (r'(\w+)\s+is\s+true.*?(\1)\s+is\s+false', "Contradictory boolean statements"),
            (r'(\w+)\s+increases.*?(\1)\s+decreases', "Contradictory trend statements"),
            (r'always.*?never', "Contradictory absolute statements"),
        ]
        
        for pattern, description in contradiction_patterns:
            if re.search(pattern, text_lower):
                contradictions.append(description)
        
        return contradictions
    
    def _addresses_task(self, output: str, task: str) -> bool:
        """Check if output addresses the task"""
        # Simple word overlap heuristic
        task_words = set(task.lower().split())
        output_words = set(output.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        task_words -= common_words
        output_words -= common_words
        
        # Check overlap
        overlap = len(task_words & output_words)
        coverage = overlap / len(task_words) if task_words else 0.0
        
        return coverage >= 0.3  # At least 30% word overlap
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics about verifications"""
        if not self.verification_history:
            return {
                'total_verifications': 0,
                'pass_rate': 0.0
            }
        
        total = len(self.verification_history)
        passed = sum(1 for v in self.verification_history if v['passed'])
        
        return {
            'total_verifications': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0.0
        }
