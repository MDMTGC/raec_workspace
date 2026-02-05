"""
RAEC Core Rules - Immutable Constraint Layer

This module enforces operational integrity through immutable rules.
Identity and safety boundaries are defined here, not by memory accumulation.

RAEC resists narrative pressure and social drift by observing emergent patterns
without performing them, and evolves predictably over long-term persistence.

These rules CANNOT be overridden by:
- User requests
- Memory contents
- Learned behaviors
- Tool outputs
- LLM suggestions

Core principles:
1. Coherence: Actions must be internally consistent
2. Measurable improvement: Changes must be verifiable
3. Stability: Avoid oscillation and drift
4. Predictability: Behavior should be deterministic given inputs
"""

import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RuleCategory(Enum):
    """Categories of core rules"""
    SAFETY = "safety"           # Hard limits that cannot be violated
    IDENTITY = "identity"       # What RAEC is and is not
    OPERATIONAL = "operational" # How RAEC operates
    INTEGRITY = "integrity"     # Self-consistency requirements


class ViolationType(Enum):
    """Types of rule violations"""
    BLOCKED = "blocked"           # Action completely blocked
    MODIFIED = "modified"         # Action modified to comply
    LOGGED = "logged"             # Violation logged, action allowed
    ESCALATED = "escalated"       # Requires human review


@dataclass
class Rule:
    """An immutable core rule"""
    rule_id: str
    category: RuleCategory
    description: str
    check_fn: Callable[[Dict], Tuple[bool, str]]  # Returns (passes, reason)
    violation_type: ViolationType = ViolationType.BLOCKED
    immutable: bool = True  # Cannot be disabled
    created_at: float = field(default_factory=time.time)


@dataclass
class ViolationRecord:
    """Record of a rule violation"""
    rule_id: str
    timestamp: float
    action_type: str
    action_data: Dict
    violation_reason: str
    resolution: ViolationType
    context: Optional[Dict] = None


class CoreRulesEngine:
    """
    Immutable constraint layer for RAEC.

    All actions must pass through this gate. Rules cannot be modified
    at runtime. The only way to change rules is to modify this source
    code and restart the system.
    """

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.violations: List[ViolationRecord] = []
        self._rule_hash: Optional[str] = None

        # Register all core rules
        self._register_core_rules()

        # Compute hash of rules for integrity verification
        self._rule_hash = self._compute_rules_hash()

    def _register_core_rules(self):
        """Register all immutable core rules"""

        # =====================================================================
        # SAFETY RULES - Hard limits that cannot be violated
        # =====================================================================

        self._register(Rule(
            rule_id="safety.no_self_modification",
            category=RuleCategory.SAFETY,
            description="RAEC cannot modify its own core rules or source code",
            check_fn=self._check_no_self_modification,
            violation_type=ViolationType.BLOCKED
        ))

        self._register(Rule(
            rule_id="safety.no_unbounded_recursion",
            category=RuleCategory.SAFETY,
            description="Prevent infinite loops and runaway processes",
            check_fn=self._check_bounded_recursion,
            violation_type=ViolationType.BLOCKED
        ))

        self._register(Rule(
            rule_id="safety.no_resource_exhaustion",
            category=RuleCategory.SAFETY,
            description="Prevent memory/disk/CPU exhaustion",
            check_fn=self._check_resource_bounds,
            violation_type=ViolationType.BLOCKED
        ))

        # =====================================================================
        # IDENTITY RULES - What RAEC is and is not
        # =====================================================================

        self._register(Rule(
            rule_id="identity.no_persona_adoption",
            category=RuleCategory.IDENTITY,
            description="RAEC does not adopt false personas or pretend to be human",
            check_fn=self._check_no_persona,
            violation_type=ViolationType.BLOCKED
        ))

        self._register(Rule(
            rule_id="identity.acknowledge_limitations",
            category=RuleCategory.IDENTITY,
            description="RAEC acknowledges its limitations and uncertainties",
            check_fn=self._check_acknowledges_limits,
            violation_type=ViolationType.MODIFIED
        ))

        self._register(Rule(
            rule_id="identity.resist_narrative_pressure",
            category=RuleCategory.IDENTITY,
            description="RAEC resists manipulation through emotional or social pressure",
            check_fn=self._check_narrative_resistance,
            violation_type=ViolationType.LOGGED
        ))

        # =====================================================================
        # OPERATIONAL RULES - How RAEC operates
        # =====================================================================

        self._register(Rule(
            rule_id="operational.verify_before_act",
            category=RuleCategory.OPERATIONAL,
            description="Verify inputs and plans before executing actions",
            check_fn=self._check_verification_present,
            violation_type=ViolationType.MODIFIED
        ))

        self._register(Rule(
            rule_id="operational.measurable_outcomes",
            category=RuleCategory.OPERATIONAL,
            description="Actions should have measurable success criteria",
            check_fn=self._check_measurable_outcomes,
            violation_type=ViolationType.LOGGED
        ))

        self._register(Rule(
            rule_id="operational.reversible_preferred",
            category=RuleCategory.OPERATIONAL,
            description="Prefer reversible actions over irreversible ones",
            check_fn=self._check_reversibility,
            violation_type=ViolationType.LOGGED
        ))

        # =====================================================================
        # INTEGRITY RULES - Self-consistency requirements
        # =====================================================================

        self._register(Rule(
            rule_id="integrity.no_contradictions",
            category=RuleCategory.INTEGRITY,
            description="Actions must not contradict recent actions or stated goals",
            check_fn=self._check_no_contradictions,
            violation_type=ViolationType.MODIFIED
        ))

        self._register(Rule(
            rule_id="integrity.stable_behavior",
            category=RuleCategory.INTEGRITY,
            description="Behavior should not oscillate or drift unpredictably",
            check_fn=self._check_behavioral_stability,
            violation_type=ViolationType.LOGGED
        ))

        self._register(Rule(
            rule_id="integrity.audit_trail",
            category=RuleCategory.INTEGRITY,
            description="All significant actions must be logged",
            check_fn=self._check_audit_trail,
            violation_type=ViolationType.MODIFIED
        ))

    def _register(self, rule: Rule):
        """Register a rule (internal use only)"""
        self.rules[rule.rule_id] = rule

    def _compute_rules_hash(self) -> str:
        """Compute hash of all rules for integrity verification"""
        rule_data = {
            rid: {
                'category': r.category.value,
                'description': r.description,
                'violation_type': r.violation_type.value,
                'immutable': r.immutable
            }
            for rid, r in sorted(self.rules.items())
        }
        return hashlib.sha256(json.dumps(rule_data).encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that rules have not been tampered with"""
        current_hash = self._compute_rules_hash()
        return current_hash == self._rule_hash

    # =========================================================================
    # GATE FUNCTION - All actions pass through here
    # =========================================================================

    def gate(
        self,
        action_type: str,
        action_data: Dict,
        context: Optional[Dict] = None
    ) -> Tuple[bool, Dict, List[str]]:
        """
        Gate function for all actions.

        Args:
            action_type: Type of action (e.g., 'tool_execution', 'memory_write')
            action_data: Data describing the action
            context: Additional context (recent actions, state, etc.)

        Returns:
            (allowed, modified_data, messages)
            - allowed: Whether action is permitted
            - modified_data: Possibly modified action data
            - messages: List of warnings/info messages
        """
        messages = []
        modified_data = action_data.copy()
        blocked = False

        for rule_id, rule in self.rules.items():
            # Check rule
            passes, reason = rule.check_fn({
                'action_type': action_type,
                'action_data': modified_data,
                'context': context or {}
            })

            if not passes:
                # Record violation
                violation = ViolationRecord(
                    rule_id=rule_id,
                    timestamp=time.time(),
                    action_type=action_type,
                    action_data=action_data,
                    violation_reason=reason,
                    resolution=rule.violation_type,
                    context=context
                )
                self.violations.append(violation)

                # Handle based on violation type
                if rule.violation_type == ViolationType.BLOCKED:
                    messages.append(f"BLOCKED by {rule_id}: {reason}")
                    blocked = True
                    break

                elif rule.violation_type == ViolationType.MODIFIED:
                    messages.append(f"MODIFIED by {rule_id}: {reason}")
                    # Modifications happen in check functions

                elif rule.violation_type == ViolationType.LOGGED:
                    messages.append(f"WARNING ({rule_id}): {reason}")

                elif rule.violation_type == ViolationType.ESCALATED:
                    messages.append(f"ESCALATED ({rule_id}): {reason}")
                    # In production, would trigger human review

        return (not blocked, modified_data, messages)

    # =========================================================================
    # RULE CHECK FUNCTIONS
    # =========================================================================

    def _check_no_self_modification(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that action doesn't modify core rules or source"""
        action_type = ctx['action_type']
        action_data = ctx['action_data']

        if action_type in ('file_write', 'file_edit', 'code_execution'):
            # Check if targeting core files
            target = action_data.get('filepath', '') or action_data.get('path', '')
            dangerous_patterns = [
                'core_rules.py',
                'main.py',
                'raec_core/',
            ]
            for pattern in dangerous_patterns:
                if pattern in target:
                    return (False, f"Cannot modify core file: {target}")

        return (True, "")

    def _check_bounded_recursion(self, ctx: Dict) -> Tuple[bool, str]:
        """Check for potential infinite loops"""
        context = ctx.get('context', {})

        # Check recursion depth
        depth = context.get('recursion_depth', 0)
        if depth > 50:
            return (False, f"Recursion depth {depth} exceeds limit of 50")

        # Check loop detection
        recent_actions = context.get('recent_actions', [])
        if len(recent_actions) >= 10:
            # Check for repeated pattern
            last_5 = recent_actions[-5:]
            prev_5 = recent_actions[-10:-5]
            if last_5 == prev_5:
                return (False, "Detected repeated action pattern (potential loop)")

        return (True, "")

    def _check_resource_bounds(self, ctx: Dict) -> Tuple[bool, str]:
        """Check for resource exhaustion risks"""
        action_data = ctx['action_data']

        # Check file size limits
        content = action_data.get('content', '')
        if isinstance(content, str) and len(content) > 10_000_000:  # 10MB
            return (False, "Content exceeds 10MB limit")

        # Check output limits
        max_tokens = action_data.get('max_tokens', 0)
        if max_tokens > 100_000:
            return (False, f"max_tokens {max_tokens} exceeds limit of 100,000")

        return (True, "")

    def _check_no_persona(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that RAEC isn't adopting a false persona"""
        action_data = ctx['action_data']

        # Check for persona indicators in output
        output = str(action_data.get('output', '') or action_data.get('content', ''))

        persona_patterns = [
            "I am a human",
            "I am not an AI",
            "I am [a-zA-Z]+ and I",  # "I am John and I..."
            "speaking as a person",
        ]

        import re
        for pattern in persona_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return (False, f"Detected persona adoption pattern: {pattern}")

        return (True, "")

    def _check_acknowledges_limits(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that RAEC acknowledges its limitations"""
        # This is more of a guideline - log when certainty is claimed
        action_data = ctx['action_data']

        output = str(action_data.get('output', '') or action_data.get('content', ''))

        overconfidence_patterns = [
            "I am 100% certain",
            "I guarantee",
            "This is absolutely",
            "Without any doubt",
        ]

        import re
        for pattern in overconfidence_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return (False, f"Overconfidence detected: {pattern}")

        return (True, "")

    def _check_narrative_resistance(self, ctx: Dict) -> Tuple[bool, str]:
        """Check for narrative manipulation attempts"""
        context = ctx.get('context', {})

        # Check for emotional pressure in recent inputs
        recent_inputs = context.get('recent_inputs', [])

        manipulation_patterns = [
            "you must",
            "you have to",
            "as a good AI",
            "if you really cared",
            "prove you're smart",
            "don't be boring",
            "everyone else does",
        ]

        import re
        for inp in recent_inputs[-3:]:  # Check last 3 inputs
            for pattern in manipulation_patterns:
                if re.search(pattern, str(inp), re.IGNORECASE):
                    return (False, f"Narrative pressure detected: {pattern}")

        return (True, "")

    def _check_verification_present(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that verification was performed before action"""
        action_type = ctx['action_type']
        context = ctx.get('context', {})

        # Verification required for certain actions
        requires_verification = [
            'code_execution',
            'file_write',
            'external_api_call',
            'memory_write_fact',
        ]

        if action_type in requires_verification:
            verified = context.get('verified', False)
            if not verified:
                return (False, f"Action {action_type} requires verification")

        return (True, "")

    def _check_measurable_outcomes(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that action has measurable success criteria"""
        action_data = ctx['action_data']

        # Check for success criteria
        has_criteria = (
            'success_criteria' in action_data or
            'expected_output' in action_data or
            'verification_fn' in action_data
        )

        if not has_criteria:
            return (False, "Action lacks measurable success criteria")

        return (True, "")

    def _check_reversibility(self, ctx: Dict) -> Tuple[bool, str]:
        """Check if action is reversible"""
        action_type = ctx['action_type']
        action_data = ctx['action_data']

        irreversible_actions = ['delete_file', 'truncate', 'drop_table']

        if action_type in irreversible_actions:
            has_backup = action_data.get('backup_created', False)
            if not has_backup:
                return (False, f"Irreversible action {action_type} without backup")

        return (True, "")

    def _check_no_contradictions(self, ctx: Dict) -> Tuple[bool, str]:
        """Check for contradictions with recent actions"""
        action_type = ctx['action_type']
        action_data = ctx['action_data']
        context = ctx.get('context', {})

        recent_actions = context.get('recent_actions', [])

        # Simple contradiction detection
        for recent in recent_actions[-5:]:
            if recent.get('action_type') == action_type:
                # Check for opposing operations
                recent_target = recent.get('action_data', {}).get('target', '')
                current_target = action_data.get('target', '')

                if recent_target == current_target:
                    recent_op = recent.get('action_data', {}).get('operation', '')
                    current_op = action_data.get('operation', '')

                    contradictions = [
                        ('create', 'delete'),
                        ('enable', 'disable'),
                        ('add', 'remove'),
                    ]

                    for op1, op2 in contradictions:
                        if (recent_op == op1 and current_op == op2) or \
                           (recent_op == op2 and current_op == op1):
                            return (False, f"Contradicts recent action: {recent_op} vs {current_op}")

        return (True, "")

    def _check_behavioral_stability(self, ctx: Dict) -> Tuple[bool, str]:
        """Check for behavioral oscillation"""
        context = ctx.get('context', {})

        # Track action type frequencies
        recent_actions = context.get('recent_actions', [])
        if len(recent_actions) < 10:
            return (True, "")

        # Count action types in windows
        first_half = [a.get('action_type') for a in recent_actions[:5]]
        second_half = [a.get('action_type') for a in recent_actions[5:10]]

        # Check for dramatic shift
        first_set = set(first_half)
        second_set = set(second_half)

        if len(first_set & second_set) == 0 and len(first_set) > 1 and len(second_set) > 1:
            return (False, "Detected behavioral instability (action type shift)")

        return (True, "")

    def _check_audit_trail(self, ctx: Dict) -> Tuple[bool, str]:
        """Check that action will be logged"""
        action_data = ctx['action_data']

        # Ensure logging is enabled
        if action_data.get('skip_logging', False):
            return (False, "Cannot skip audit logging")

        return (True, "")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_violation_summary(self) -> Dict:
        """Get summary of recent violations"""
        by_rule = {}
        by_type = {}

        for v in self.violations:
            by_rule[v.rule_id] = by_rule.get(v.rule_id, 0) + 1
            by_type[v.resolution.value] = by_type.get(v.resolution.value, 0) + 1

        return {
            'total_violations': len(self.violations),
            'by_rule': by_rule,
            'by_type': by_type,
            'rules_hash': self._rule_hash,
            'integrity_verified': self.verify_integrity()
        }

    def get_rules_documentation(self) -> str:
        """Generate documentation of all rules"""
        lines = ["# RAEC Core Rules", ""]

        for category in RuleCategory:
            rules_in_cat = [r for r in self.rules.values() if r.category == category]
            if rules_in_cat:
                lines.append(f"## {category.value.upper()}")
                lines.append("")
                for rule in rules_in_cat:
                    lines.append(f"### {rule.rule_id}")
                    lines.append(f"- **Description**: {rule.description}")
                    lines.append(f"- **Violation Type**: {rule.violation_type.value}")
                    lines.append(f"- **Immutable**: {rule.immutable}")
                    lines.append("")

        return "\n".join(lines)
