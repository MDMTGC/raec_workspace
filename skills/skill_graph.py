"""
Audited Skill Graph - Self-Improvement Module
Based on ASG-SI (Audited Skill-Graph Self-Improvement) architecture

Treats agent learning as skill extraction and verification, producing
auditable, reusable capabilities driven by verifiable rewards.
"""
import json
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class SkillStatus(Enum):
    """Verification status of a skill"""
    CANDIDATE = "candidate"      # Newly extracted, unverified
    VERIFIED = "verified"        # Passed verification tests
    DEPRECATED = "deprecated"    # Superseded by better skill
    FAILED = "failed"            # Failed verification


class SkillCategory(Enum):
    """Types of skills"""
    TOOL_USAGE = "tool_usage"           # Using external tools
    REASONING = "reasoning"             # Logic and inference patterns
    PLANNING = "planning"               # Task decomposition strategies
    DATA_PROCESSING = "data_processing" # Data transformation patterns
    CODE_GENERATION = "code_generation" # Code writing patterns
    INTEGRATION = "integration"         # System integration patterns


@dataclass
class SkillEvidence:
    """Evidence supporting a skill's validity"""
    task_description: str
    execution_result: Dict[str, Any]
    success_metrics: Dict[str, float]
    timestamp: float
    memory_id: Optional[int] = None


@dataclass
class SkillVerification:
    """Verification test results for a skill"""
    test_description: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: float


@dataclass
class Skill:
    """
    A verified, reusable capability
    """
    skill_id: str
    name: str
    description: str
    category: SkillCategory
    solution_pattern: str  # Template or pattern for solving tasks
    prerequisites: List[str]  # Required skills or knowledge
    evidence: List[SkillEvidence]
    verifications: List[SkillVerification]
    status: SkillStatus
    confidence: float  # 0-1 based on verification results
    usage_count: int
    success_rate: float
    created_at: float
    last_used: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['category'] = self.category.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Skill':
        """Create from dictionary"""
        # Convert string enums back
        data['category'] = SkillCategory(data['category'])
        data['status'] = SkillStatus(data['status'])
        
        # Reconstruct nested dataclasses
        data['evidence'] = [
            SkillEvidence(**e) if isinstance(e, dict) else e 
            for e in data.get('evidence', [])
        ]
        data['verifications'] = [
            SkillVerification(**v) if isinstance(v, dict) else v 
            for v in data.get('verifications', [])
        ]
        
        return cls(**data)


class SkillGraph:
    """
    Audited Skill Graph for self-improvement
    
    Features:
    - Extract skills from successful task completions
    - Verify skills with test cases
    - Build dependency graph between skills
    - Track performance and evidence
    - Audit trail for all skill changes
    """
    
    def __init__(self, storage_path: str = "skills/skill_db.json"):
        self.storage_path = storage_path
        self.skills: Dict[str, Skill] = {}
        self.skill_dependencies: Dict[str, List[str]] = {}  # skill_id -> [dependency_ids]
        self.audit_log: List[Dict] = []
        
        self._load()
    
    def extract_skill(
        self,
        task_description: str,
        solution: str,
        execution_result: Dict[str, Any],
        category: SkillCategory,
        name: Optional[str] = None,
        prerequisites: Optional[List[str]] = None
    ) -> str:
        """
        Extract a new skill from successful task completion
        
        Args:
            task_description: What task was solved
            solution: How it was solved (code, steps, pattern)
            execution_result: Results of execution
            category: Type of skill
            name: Optional skill name
            prerequisites: Required skills/knowledge
            
        Returns:
            skill_id of extracted skill
        """
        # Generate unique skill ID
        skill_id = self._generate_skill_id(task_description, solution)
        
        # Check if similar skill already exists
        similar = self._find_similar_skills(task_description, category)
        if similar:
            # Check if any similar skill is already verified
            verified_similar = [s for s in similar if s.status == SkillStatus.VERIFIED]
            if verified_similar:
                existing = verified_similar[0]
                print(f"[!] Similar verified skill already exists: {existing.name}")
                print(f"    Skipping extraction - use existing skill instead")
                # Add this execution as additional evidence to existing skill
                evidence = SkillEvidence(
                    task_description=task_description,
                    execution_result=execution_result,
                    success_metrics=self._compute_success_metrics(execution_result),
                    timestamp=time.time()
                )
                existing.evidence.append(evidence)
                self._save()
                return existing.skill_id
            else:
                print(f"[!] Similar unverified skill exists: {similar[0].name}")
                # Continue with extraction - may supersede the unverified one
        
        # Create evidence from execution
        evidence = SkillEvidence(
            task_description=task_description,
            execution_result=execution_result,
            success_metrics=self._compute_success_metrics(execution_result),
            timestamp=time.time()
        )
        
        # Create skill as CANDIDATE (needs verification)
        skill = Skill(
            skill_id=skill_id,
            name=name or self._generate_skill_name(task_description),
            description=task_description,
            category=category,
            solution_pattern=solution,
            prerequisites=prerequisites or [],
            evidence=[evidence],
            verifications=[],
            status=SkillStatus.CANDIDATE,
            confidence=0.5,  # Initial confidence
            usage_count=0,
            success_rate=0.0,
            created_at=time.time(),
            metadata={}
        )
        
        self.skills[skill_id] = skill
        
        self._log_audit("skill_extracted", {
            'skill_id': skill_id,
            'name': skill.name,
            'category': category.value
        })
        
        print(f"[+] Extracted new skill: {skill.name} (ID: {skill_id[:8]}...)")
        print(f"    Status: CANDIDATE - needs verification")
        
        self._save()
        return skill_id
    
    def verify_skill(
        self,
        skill_id: str,
        test_cases: List[Dict[str, Any]],
        verifier_fn: Optional[callable] = None
    ) -> bool:
        """
        Verify a skill against test cases
        
        Args:
            skill_id: Skill to verify
            test_cases: List of test scenarios
            verifier_fn: Optional custom verification function
            
        Returns:
            True if skill passes verification
        """
        if skill_id not in self.skills:
            raise ValueError(f"Skill {skill_id} not found")
        
        skill = self.skills[skill_id]
        
        print(f"[?] Verifying skill: {skill.name}")
        print(f"   Running {len(test_cases)} test cases...")
        
        passed_tests = 0
        verifications = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"   Test {i}/{len(test_cases)}: {test_case.get('description', 'Unnamed test')}", end=" ")
            
            # Run verification
            if verifier_fn:
                result = verifier_fn(skill, test_case)
            else:
                result = self._default_verifier(skill, test_case)
            
            verification = SkillVerification(
                test_description=test_case.get('description', f'Test {i}'),
                passed=result['passed'],
                score=result['score'],
                details=result.get('details', {}),
                timestamp=time.time()
            )
            
            verifications.append(verification)
            skill.verifications.append(verification)
            
            if result['passed']:
                passed_tests += 1
                print("[OK]")
            else:
                print("[X]")
        
        # Calculate overall pass rate
        pass_rate = passed_tests / len(test_cases)
        
        # Update skill status based on verification
        if pass_rate >= 0.8:  # 80% pass threshold
            skill.status = SkillStatus.VERIFIED
            skill.confidence = pass_rate
            print(f"\n   [OK] VERIFIED - Pass rate: {pass_rate:.1%}")
        else:
            skill.status = SkillStatus.FAILED
            skill.confidence = pass_rate
            print(f"\n   [FAIL] FAILED - Pass rate: {pass_rate:.1%} (need 80%)")
        
        self._log_audit("skill_verified", {
            'skill_id': skill_id,
            'status': skill.status.value,
            'pass_rate': pass_rate,
            'num_tests': len(test_cases)
        })
        
        self._save()
        return skill.status == SkillStatus.VERIFIED
    
    def _default_verifier(self, skill: Skill, test_case: Dict) -> Dict:
        """
        Default verification logic

        Supports multiple test case formats:
        1. {'expected': True/False} - declarative pass/fail
        2. {'expected': value, 'actual': value} - value comparison
        3. {'error': None} - pass if no error
        4. {'pass': True/False} - explicit pass/fail
        """
        expected = test_case.get('expected', None)
        actual = test_case.get('actual', None)
        explicit_pass = test_case.get('pass', None)

        # Format 4: Explicit pass/fail
        if explicit_pass is not None:
            passed = bool(explicit_pass)
            score = 1.0 if passed else 0.0
            return {'passed': passed, 'score': score, 'details': {'mode': 'explicit'}}

        # Format 1: Declarative - expected is a boolean indicating should-pass
        if expected is not None and actual is None and isinstance(expected, bool):
            passed = expected
            score = 1.0 if passed else 0.0
            return {'passed': passed, 'score': score, 'details': {'mode': 'declarative'}}

        # Format 3: No expected, check for error
        if expected is None:
            passed = test_case.get('error') is None
            score = 1.0 if passed else 0.0
            return {'passed': passed, 'score': score, 'details': {'mode': 'error_check'}}

        # Format 2: Compare expected vs actual
        passed = expected == actual
        score = 1.0 if passed else 0.0
        return {
            'passed': passed,
            'score': score,
            'details': {
                'mode': 'comparison',
                'expected': expected,
                'actual': actual
            }
        }
    
    def query_skill(self, task_description: str, category: Optional[SkillCategory] = None) -> Optional[Skill]:
        """
        Find a verified skill that matches the task
        
        Args:
            task_description: Task to find skill for
            category: Optional category filter
            
        Returns:
            Best matching verified skill or None
        """
        # Filter to verified skills only
        verified_skills = [
            s for s in self.skills.values() 
            if s.status == SkillStatus.VERIFIED
        ]
        
        if category:
            verified_skills = [s for s in verified_skills if s.category == category]
        
        if not verified_skills:
            return None
        
        # Simple matching based on description overlap
        # In production, would use embeddings
        best_match = None
        best_score = 0.0
        
        task_words = set(task_description.lower().split())
        
        for skill in verified_skills:
            skill_words = set(skill.description.lower().split())
            overlap = len(task_words & skill_words)
            score = overlap / max(len(task_words), len(skill_words))
            
            if score > best_score:
                best_score = score
                best_match = skill
        
        # Threshold for match (lowered to allow partial matches)
        if best_score >= 0.15:
            return best_match

        return None
    
    def has_skill(self, task_description: str) -> bool:
        """Check if a verified skill exists for this task"""
        return self.query_skill(task_description) is not None
    
    def use_skill(self, skill_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a skill to a new context
        
        Args:
            skill_id: Skill to use
            context: Context/parameters for application
            
        Returns:
            Result of applying the skill
        """
        if skill_id not in self.skills:
            raise ValueError(f"Skill {skill_id} not found")
        
        skill = self.skills[skill_id]
        
        if skill.status != SkillStatus.VERIFIED:
            print(f"[!] Warning: Using unverified skill {skill.name}")
        
        # Update usage stats
        skill.usage_count += 1
        skill.last_used = time.time()
        
        self._log_audit("skill_used", {
            'skill_id': skill_id,
            'context': context
        })
        
        # Apply skill (placeholder - actual implementation depends on skill type)
        result = {
            'skill_id': skill_id,
            'skill_name': skill.name,
            'solution_pattern': skill.solution_pattern,
            'applied': True
        }
        
        self._save()
        return result
    
    def record_skill_outcome(
        self,
        skill_id: str,
        success: bool,
        result: Optional[Dict] = None
    ):
        """Record the outcome of using a skill"""
        if skill_id not in self.skills:
            return
        
        skill = self.skills[skill_id]
        
        # Update success rate
        total_uses = skill.usage_count
        if total_uses > 0:
            current_successes = skill.success_rate * (total_uses - 1)
            new_successes = current_successes + (1 if success else 0)
            skill.success_rate = new_successes / total_uses
        else:
            skill.success_rate = 1.0 if success else 0.0
        
        # If skill is performing poorly, deprecate it
        if skill.usage_count >= 10 and skill.success_rate < 0.5:
            skill.status = SkillStatus.DEPRECATED
            print(f"[!] Skill {skill.name} deprecated due to low success rate ({skill.success_rate:.1%})")
        
        self._save()
    
    def add_skill_dependency(self, skill_id: str, depends_on: str):
        """Add a dependency between skills"""
        if skill_id not in self.skill_dependencies:
            self.skill_dependencies[skill_id] = []
        
        if depends_on not in self.skill_dependencies[skill_id]:
            self.skill_dependencies[skill_id].append(depends_on)
            self._save()
    
    def get_skill_chain(self, skill_id: str) -> List[Skill]:
        """Get all skills in the dependency chain"""
        chain = []
        visited = set()
        
        def traverse(sid):
            if sid in visited or sid not in self.skills:
                return
            visited.add(sid)
            chain.append(self.skills[sid])
            
            for dep in self.skill_dependencies.get(sid, []):
                traverse(dep)
        
        traverse(skill_id)
        return chain
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the skill graph"""
        total_skills = len(self.skills)
        
        by_status = {}
        by_category = {}
        
        for skill in self.skills.values():
            # Count by status
            status = skill.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by category
            category = skill.category.value
            by_category[category] = by_category.get(category, 0) + 1
        
        verified_skills = [s for s in self.skills.values() if s.status == SkillStatus.VERIFIED]
        avg_confidence = sum(s.confidence for s in verified_skills) / len(verified_skills) if verified_skills else 0.0
        total_usage = sum(s.usage_count for s in self.skills.values())
        
        return {
            'total_skills': total_skills,
            'by_status': by_status,
            'by_category': by_category,
            'verified_count': len(verified_skills),
            'avg_confidence': avg_confidence,
            'total_usage': total_usage
        }
    
    def _generate_skill_id(self, task: str, solution: str) -> str:
        """Generate unique ID for a skill"""
        content = f"{task}:{solution}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_skill_name(self, task_description: str) -> str:
        """Generate a readable name from task description"""
        # Strip common system directive patterns
        cleaned = task_description

        # Remove "You are a..." preambles
        import re
        cleaned = re.sub(r'^You are [^.]+\.\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove "Task:" prefix
        cleaned = re.sub(r'^Task:\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove role/instruction preambles
        cleaned = re.sub(r'^(Provide|Generate|Create|Analyze|Execute)[^.]*\.\s*', '', cleaned, flags=re.IGNORECASE)

        # Take first meaningful words
        words = cleaned.strip().split()[:5]
        name = " ".join(words)

        # Fallback if empty
        if not name or len(name) < 3:
            words = task_description.split()[:5]
            name = " ".join(words)

        if len(name) > 50:
            name = name[:47] + "..."
        return name
    
    def _find_similar_skills(self, task: str, category: SkillCategory) -> List[Skill]:
        """Find skills similar to the given task"""
        # Simplified similarity - in production would use embeddings
        similar = []
        task_words = set(task.lower().split())
        
        for skill in self.skills.values():
            if skill.category != category:
                continue
            
            skill_words = set(skill.description.lower().split())
            overlap = len(task_words & skill_words)
            
            if overlap >= 3:  # At least 3 words in common
                similar.append(skill)
        
        return similar
    
    def _compute_success_metrics(self, execution_result: Dict) -> Dict[str, float]:
        """Extract success metrics from execution result"""
        metrics = {
            'success': 1.0 if execution_result.get('success', False) else 0.0
        }
        
        # Add any other metrics from result
        if 'metrics' in execution_result:
            metrics.update(execution_result['metrics'])
        
        return metrics
    
    def _log_audit(self, action: str, details: Dict):
        """Add entry to audit log"""
        self.audit_log.append({
            'timestamp': time.time(),
            'action': action,
            'details': details
        })
        
        # Keep only recent audit entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def _save(self):
        """Persist skill graph to disk"""
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'skills': {sid: skill.to_dict() for sid, skill in self.skills.items()},
                'dependencies': self.skill_dependencies,
                'audit_log': self.audit_log[-100:]  # Save recent audit entries
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[!] Failed to save skill graph: {e}")
    
    def _load(self):
        """Load skill graph from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.skills = {
                sid: Skill.from_dict(skill_data) 
                for sid, skill_data in data.get('skills', {}).items()
            }
            self.skill_dependencies = data.get('dependencies', {})
            self.audit_log = data.get('audit_log', [])
            
            print(f"[OK] Loaded {len(self.skills)} skills from {self.storage_path}")
        except FileNotFoundError:
            print(f"[OK] Initialized new skill graph at {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load skill graph: {e}")
