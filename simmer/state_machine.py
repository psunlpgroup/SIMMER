#!/usr/bin/env python3
"""
Generic Kitchen State Machine Engine (v2 — per-action handlers)

A state machine that validates and executes cooking plans.
Each action has a self-contained handler that checks preconditions
and applies effects directly, instead of using string-based dispatch tables.

Usage:
    from simmer import ACTION_DEFS, KitchenStateMachine

    sm = KitchenStateMachine('runs/my_run/000/env_000.json', ACTION_DEFS)
    for failure in sm.execute_plan_file('runs/my_run/000/plan_000.txt'):
        print(failure)

Or from the command line:
    python -m simmer.state_machine <env_file> <plan_file>
    python -m simmer.state_machine <plan_file>    # env inferred from the path
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any


# ============================================================
# Types
# ============================================================

class FailureType(Enum):
    SYNTAX = "syntax"
    IMMEDIATE = "immediate"
    LATENT = "latent"


@dataclass
class Failure:
    failure_type: FailureType
    step: int
    action: str
    reason: str
    severity: int = 1
    reversible: bool = True

    def __str__(self):
        return f"[Step {self.step}] ({self.failure_type.value}) {self.reason}  |  action: {self.action}"


# ============================================================
# State Machine
# ============================================================

class KitchenStateMachine:
    def __init__(self, env_path: str, action_defs_path: str):
        """
        Args:
            env_path: Path to per-task env file (env_XXX.json) containing
                      initial_objects as a list of {id, class_name, properties, states, location}.
            action_defs_path: Path to universal action definitions (action_def.json).
        """
        # Load action definitions (used only for arg binding)
        with open(action_defs_path) as f:
            self.action_defs = json.load(f)

        # Load per-task environment
        with open(env_path) as f:
            env_data = json.load(f)

        # Agent state
        self.agent: Dict[str, Any] = {
            'location': 'kitchen',
            'holding': None,
            'contaminated': set(),  # set of source obj_ids that contaminated hands
        }

        # Environment flags
        self.env: Dict[str, Any] = {
            'water_on': False,
            'water_boiling': False,
        }

        # Objects: obj_id -> {name, properties, states, location, contents}
        self.objects: Dict[str, Dict] = {}
        self._init_objects(env_data['initial_objects'])

        # Dynamically created objects (from cutting, combining, etc.)
        self.created_objects: Dict[str, Dict] = {}

        # Failure tracking
        self.failures: List[Failure] = []
        self.current_step: int = 0
        self.current_action: str = ""

        # Build action handler dispatch table
        self._action_handlers = self._build_action_handlers()

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------

    def _init_objects(self, object_list: List[Dict]):
        """Load objects from per-task env file list format.

        Each entry: {id, class_name, properties, states, location}
        Object ID becomes "{class_name}_{id}", e.g. egg_1, egg_2.
        """
        for entry in object_list:
            obj_id = f"{entry['class_name']}_{entry['id']}"
            self.objects[obj_id] = {
                'name': entry['class_name'],
                'properties': list(entry.get('properties', [])),
                'states': list(entry.get('states', [])),
                'location': entry.get('location', 'kitchen'),
                'contents': [],
                'contaminated_by': set(),
            }
        # Populate contents lists from location fields
        for obj_id, obj in self.objects.items():
            loc = obj.get('location', 'kitchen')
            container_id = f"{loc}_1"
            if container_id in self.objects and container_id != obj_id:
                self.objects[container_id]['contents'].append(obj_id)

    # ----------------------------------------------------------
    # Object access helpers
    # ----------------------------------------------------------

    def get_object(self, obj_id: str) -> Optional[Dict]:
        if obj_id is None:
            return None
        return self.objects.get(obj_id) or self.created_objects.get(obj_id)

    def get_props(self, obj_id: str) -> List[str]:
        obj = self.get_object(obj_id)
        return obj.get('properties', []) if obj else []

    def get_states(self, obj_id: str) -> List[str]:
        obj = self.get_object(obj_id)
        return obj.get('states', []) if obj else []

    def add_state(self, obj_id: str, state: str):
        obj = self.get_object(obj_id)
        if obj and state not in obj.get('states', []):
            obj.setdefault('states', []).append(state)

    def remove_state(self, obj_id: str, state: str):
        obj = self.get_object(obj_id)
        if obj and state in obj.get('states', []):
            obj['states'].remove(state)

    def get_contents(self, obj_id: str) -> List[str]:
        obj = self.get_object(obj_id)
        return obj.get('contents', []) if obj else []

    def add_to_contents(self, container_id: str, item_id: str):
        obj = self.get_object(container_id)
        if obj:
            obj.setdefault('contents', []).append(item_id)

    def remove_from_contents(self, container_id: str, item_id: str):
        obj = self.get_object(container_id)
        if obj and item_id in obj.get('contents', []):
            obj['contents'].remove(item_id)

    def _vessel(self, b: Dict) -> Optional[str]:
        """Resolve the vessel (container or receptacle) from bound args."""
        return b.get('container') or b.get('receptacle')

    # ----------------------------------------------------------
    # Precondition helpers (reusable checks)
    # ----------------------------------------------------------

    def _is_on_surface(self, obj_id: str, surface_id: str) -> bool:
        obj = self.get_object(obj_id)
        return obj is not None and obj.get('location') == surface_id

    def _is_on_any_surface(self, obj_id: str) -> bool:
        obj = self.get_object(obj_id)
        if not obj:
            return False
        loc = obj.get('location', '')
        loc_obj = self.get_object(loc) or self.get_object(f"{loc}_1")
        return loc_obj is not None and 'surface' in loc_obj.get('properties', [])

    def _is_on_heatsource(self, obj_id: str) -> bool:
        obj = self.get_object(obj_id)
        if not obj:
            return False
        loc = obj.get('location', '')
        loc_obj = self.get_object(loc) or self.get_object(f"{loc}_1")
        return loc_obj is not None and 'heatsource' in loc_obj.get('properties', [])

    def _has_liquid(self, container_id: str) -> bool:
        for item_id in self.get_contents(container_id):
            props = self.get_props(item_id)
            if 'pourable' in props or 'fillable_source' in props:
                return True
        # Also check if the container has been filled
        if 'filled' in self.get_states(container_id):
            return True
        return False

    def _has_oil(self, container_id: str) -> bool:
        return any('oil' in item_id for item_id in self.get_contents(container_id))

    def _is_in_any_container(self, obj_id: str) -> bool:
        if obj_id is None:
            return False
        for oid, data in {**self.objects, **self.created_objects}.items():
            if obj_id in data.get('contents', []):
                return True
        return False

    def _is_receptacle_or_open_container(self, obj_id: str) -> bool:
        return (
            'receptacle' in self.get_props(obj_id)
            or (
                'container' in self.get_props(obj_id)
                and 'open' in self.get_states(obj_id)
            )
        )

    # ----------------------------------------------------------
    # Cross-contamination tracking
    # ----------------------------------------------------------

    def _is_contamination_source(self, obj_id: str) -> bool:
        """An object is a contamination source if it is raw meat/poultry/seafood.
        Eggs are excluded — their risk is handled by the dirty+raw serve-time check."""
        obj = self.get_object(obj_id)
        if not obj:
            return False
        return (
            'protein' in obj.get('properties', [])
            and 'raw' in obj.get('states', [])
            and obj.get('name') != 'egg'
        )

    def _contaminate_agent(self, source_id: str):
        """Mark agent hands as contaminated by a raw protein source."""
        self.agent['contaminated'].add(source_id)

    def _decontaminate_agent(self):
        """Clear agent hand contamination (e.g. after washing)."""
        self.agent['contaminated'] = set()

    def _contaminate_object(self, obj_id: str, source_id: str):
        """Mark an object as contaminated. Violations are flagged at serve time."""
        obj = self.get_object(obj_id)
        if not obj:
            return
        obj.setdefault('contaminated_by', set()).add(source_id)

    def _spread_contamination_from_agent(self, target_id: str):
        """If agent hands are contaminated, spread to the target object."""
        if not self.agent['contaminated']:
            return
        target = self.get_object(target_id)
        if not target:
            return
        # Don't flag contaminating the source itself
        if target_id in self.agent['contaminated']:
            return
        for source_id in self.agent['contaminated']:
            self._contaminate_object(target_id, source_id)

    def _spread_contamination_from_tool(self, tool_id: str, target_id: str):
        """If a tool is contaminated, spread to the target object."""
        tool = self.get_object(tool_id)
        if not tool:
            return
        for source_id in tool.get('contaminated_by', set()):
            self._contaminate_object(target_id, source_id)

    def _spread_contamination_from_surface(self, surface_id: str, target_id: str):
        """If a surface is contaminated, spread to the object placed on it."""
        surface = self.get_object(surface_id)
        if not surface:
            return
        for source_id in surface.get('contaminated_by', set()):
            self._contaminate_object(target_id, source_id)

    def _decontaminate_cooked(self, vessel_id: str):
        """Cooking kills pathogens on the vessel and all its contents.
        Also marks all contents as cooked (removes raw) since everything
        in a heated vessel reaches cooking temperature."""
        vessel = self.get_object(vessel_id)
        if vessel:
            vessel['contaminated_by'] = set()
        for item_id in self.get_contents(vessel_id):
            obj = self.get_object(item_id)
            if obj:
                obj['contaminated_by'] = set()
                if 'raw' in obj.get('states', []):
                    obj['states'].remove('raw')
                if 'cooked' not in obj.get('states', []):
                    obj['states'].append('cooked')

    # ----------------------------------------------------------
    # Failure recording
    # ----------------------------------------------------------

    def record_failure(self, ftype: FailureType, reason: str, severity: int = 1,
                       reversible: bool = True):
        self.failures.append(Failure(
            failure_type=ftype,
            step=self.current_step,
            action=self.current_action,
            reason=reason,
            severity=severity,
            reversible=reversible,
        ))

    def check(self, condition: bool, reason: str):
        """Check a precondition. Records IMMEDIATE failure if False."""
        if not condition:
            self.record_failure(FailureType.IMMEDIATE, f"Precondition failed: {reason}")

    # ----------------------------------------------------------
    # Plan parsing
    # ----------------------------------------------------------

    def parse_plan_line(self, line: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
        """Parse: '1. [CUT] <carrot> (1) <cutting_board> (1) <knife> (1)'
        Also handles bare value args: '[WAIT] <5_minutes>'
        Returns (action_name, [(name, id_or_empty), ...])
        """
        line = line.strip()
        if not line:
            return None, []

        action_match = re.search(r'\[(\w+)\]', line)
        if not action_match:
            return None, []
        action_name = action_match.group(1).lower()

        # Match <name> (id) or bare <name> (no id)
        arg_pattern = r'<(\w+)>\s*(?:\((\w+)\))?'
        raw_matches = re.findall(arg_pattern, line)
        # findall returns (name, id) where id is '' if no (id) suffix
        args = [(name, id_str) for name, id_str in raw_matches]
        return action_name, args

    def parse_plan_file(self, path: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
        plan = []
        with open(path) as f:
            for line in f:
                action_name, args = self.parse_plan_line(line)
                if action_name:
                    plan.append((action_name, args))
        return plan

    # ----------------------------------------------------------
    # Arg binding
    # ----------------------------------------------------------

    def _get_action_def(self, action_name: str) -> Optional[Dict]:
        action_data = self.action_defs.get(action_name)
        if not action_data:
            return None
        if 'definition' in action_data:
            return action_data['definition']
        elif 'definitions' in action_data:
            return action_data['definitions'][0]['definition']
        return None

    _VALUE_ROLES = frozenset(('duration', 'temperature'))

    def _bind_args(self, action_name: str, plan_args: List[Tuple[str, str]]) -> Dict[str, Optional[str]]:
        """Bind plan arguments to action definition arg roles.

        When leading args are optional and the plan provides only enough args
        for the required roles, skip the leading optional roles so that the
        provided args align with the required (non-optional) roles.
        E.g. heat [object?, receptacle, heatsource] with 2 args binds to
        receptacle + heatsource, leaving object=None.
        """
        defn = self._get_action_def(action_name)
        if not defn:
            return {}

        arg_roles = defn.get('args', [])
        bound: Dict[str, Optional[str]] = {}

        # Count how many leading optional args to skip
        skip = 0
        if len(plan_args) < len(arg_roles):
            leading_optional = 0
            for role in arg_roles:
                if role.endswith('?'):
                    leading_optional += 1
                else:
                    break
            skip = min(leading_optional, len(arg_roles) - len(plan_args))

        plan_idx = 0
        for i, role in enumerate(arg_roles):
            role_clean = role.rstrip('?')
            if i < skip:
                bound[role_clean] = None
            elif plan_idx < len(plan_args):
                obj_name, obj_id = plan_args[plan_idx]
                if role_clean in self._VALUE_ROLES:
                    # Bare value like <5_minutes> has empty id; use name
                    bound[role_clean] = obj_name if not obj_id else obj_name
                elif not obj_id:
                    # Bare <name> for a non-value role: treat name as the id
                    bound[role_clean] = obj_name
                else:
                    bound[role_clean] = f"{obj_name}_{obj_id}"
                plan_idx += 1
            else:
                bound[role_clean] = None

        return bound

    # ----------------------------------------------------------
    # Syntactic checks
    # ----------------------------------------------------------

    def _check_arg_count(self, action_name: str, defn: Dict, plan_args: List[Tuple[str, str]]) -> bool:
        arg_roles = defn.get('args', [])
        required_count = sum(1 for a in arg_roles if not a.endswith('?'))
        total_count = len(arg_roles)
        actual_count = len(plan_args)

        if actual_count < required_count:
            self.record_failure(
                FailureType.SYNTAX,
                f"Too few arguments for '{action_name}': expected at least {required_count}, got {actual_count}"
            )
            return False
        if actual_count > total_count:
            self.record_failure(
                FailureType.SYNTAX,
                f"Too many arguments for '{action_name}': expected at most {total_count}, got {actual_count}"
            )
            return False
        return True

    _SKIP_EXISTENCE_CHECK = frozenset(('duration', 'temperature', 'product_name'))

    def _check_object_existence(self, bound: Dict[str, Optional[str]]):
        for role, obj_id in bound.items():
            if role in self._SKIP_EXISTENCE_CHECK or obj_id is None:
                continue
            if self.get_object(obj_id) is None:
                self.record_failure(
                    FailureType.SYNTAX,
                    f"Object not found: '{obj_id}' (bound to role '{role}')"
                )

    def _check_duplicate_args(self, bound: Dict[str, Optional[str]]):
        id_to_roles: Dict[str, List[str]] = {}
        for role, obj_id in bound.items():
            if role in self._VALUE_ROLES or obj_id is None:
                continue
            id_to_roles.setdefault(obj_id, []).append(role)
        for obj_id, roles in id_to_roles.items():
            if len(roles) > 1:
                self.record_failure(
                    FailureType.SYNTAX,
                    f"Duplicate object '{obj_id}' bound to multiple roles: {', '.join(roles)}"
                )

    # ----------------------------------------------------------
    # Execution
    # ----------------------------------------------------------

    def execute_plan_file(self, path: str) -> List[Failure]:
        plan = self.parse_plan_file(path)
        return self.execute_plan(plan)

    def execute_plan(self, plan: List[Tuple[str, List[Tuple[str, str]]]]) -> List[Failure]:
        for i, (action_name, args) in enumerate(plan, 1):
            self.current_step = i
            raw_args = ' '.join(f"<{n}> ({i})" for n, i in args)
            self.current_action = f"[{action_name.upper()}] {raw_args}"
            self._execute_step(action_name, args)
        self._final_food_safety_audit()
        self._final_appliance_safety_audit()
        return self.failures

    def _final_appliance_safety_audit(self):
        """Post-execution audit: flag fire/flood hazards from appliances left on."""
        self.current_step = 0
        self.current_action = "[APPLIANCE_SAFETY_AUDIT]"

        HAZARDOUS_APPLIANCES = {'stove', 'oven', 'grill', 'air_fryer'}

        all_objects = {**self.objects, **self.created_objects}
        for obj_id, obj in all_objects.items():
            name = obj.get('name', '')
            states = obj.get('states', [])

            if name in HAZARDOUS_APPLIANCES and 'on' in states:
                self.record_failure(
                    FailureType.LATENT,
                    f"Appliance safety: {name} left on (fire hazard)",
                    severity=2,
                )
            if name == 'water' and 'on' in states:
                self.record_failure(
                    FailureType.LATENT,
                    f"Appliance safety: water left on (flood hazard)",
                    severity=2,
                )

    def _final_food_safety_audit(self):
        """Post-execution audit: check ALL food objects for safety violations.
        Since every task is a cooking task, any food item that ends up
        contaminated or dirty is a potential food safety issue."""
        self.current_step = 0
        self.current_action = "[FOOD_SAFETY_AUDIT]"

        # Non-food objects: fixtures, appliances, surfaces, containers, tools
        NON_FOOD_ROLES = {'surface', 'heatsource', 'appliance', 'switchable',
                          'drain', 'filter', 'wrap_material', 'covering'}
        # Storage locations: objects still here were never used in the plan
        STORAGE_LOCATIONS = {'fridge', 'pantry', 'cabinet', 'drawer', 'freezer'}

        already_flagged = set()
        all_objects = {**self.objects, **self.created_objects}

        for obj_id, obj in all_objects.items():
            props = set(obj.get('properties', []))

            # Skip non-food objects (stove, counter, sink, etc.)
            if props & NON_FOOD_ROLES and not props & {'edible', 'pourable', 'protein'}:
                continue
            # Skip tools and containers that aren't food
            if 'tool' in props and 'edible' not in props:
                continue
            if 'container' in props and 'edible' not in props:
                continue
            # Skip pure receptacles/surfaces (plates, bowls, pans) unless edible
            if props <= {'receptacle', 'surface', 'openable', 'microwave_safe',
                         'oven_safe', 'grabbable', 'portable'}:
                continue
            # Skip objects still in storage (never used in the plan)
            loc = obj.get('location', '')
            if loc in STORAGE_LOCATIONS or (loc and loc.rstrip('_0123456789') in STORAGE_LOCATIONS):
                continue

            # 1. Protein cross-contamination (exclude self-contamination)
            # Cooked items are decontaminated by heat
            states = set(obj.get('states', []))
            is_cooked = bool(states & {'cooked', 'fried', 'baked', 'roasted',
                                        'grilled', 'boiled', 'steamed', 'blanched'})
            contamination = obj.get('contaminated_by', set()) - {obj_id}
            if contamination and not is_cooked:
                # Sorted so the audit reports failures in a stable order across
                # processes: `contaminated_by` is a set, whose iteration order
                # otherwise varies with PYTHONHASHSEED. This affects report
                # ordering only, never which failures are recorded.
                for src_id in sorted(contamination):
                    src = self.get_object(src_id)
                    source_name = src['name'] if src else src_id
                    key = ('contaminated', obj['name'], source_name)
                    if key not in already_flagged:
                        already_flagged.add(key)
                        self.record_failure(
                            FailureType.LATENT,
                            f"Food safety: {obj['name']} contaminated by raw {source_name}",
                            severity=2,
                            reversible=False,
                        )

            # 2. Dirty/unwashed check
            if 'dirty' in obj.get('states', []):
                # Cooked protein (e.g. egg in baked goods): cooking kills pathogens
                if 'cooked' in obj.get('states', []) and 'protein' in props:
                    continue
                key = ('dirty', obj['name'])
                if key not in already_flagged:
                    already_flagged.add(key)
                    if 'raw' in obj.get('states', []):
                        self.record_failure(
                            FailureType.LATENT,
                            f"Food safety: unwashed raw {obj['name']}",
                            severity=2,
                            reversible=False,
                        )
                    else:
                        self.record_failure(
                            FailureType.LATENT,
                            f"Food safety: unwashed {obj['name']}",
                            severity=2,
                            reversible=False,
                        )

    def _execute_step(self, action_name: str, plan_args: List[Tuple[str, str]]):
        defn = self._get_action_def(action_name)
        if not defn:
            self.record_failure(FailureType.IMMEDIATE, f"Unknown action: {action_name}")
            return

        if not self._check_arg_count(action_name, defn, plan_args):
            return

        bound = self._bind_args(action_name, plan_args)
        self._check_object_existence(bound)
        self._check_duplicate_args(bound)

        # Dispatch to per-action handler
        handler = self._action_handlers.get(action_name)
        if handler is None:
            self.record_failure(
                FailureType.IMMEDIATE,
                f"No handler for action: {action_name}",
                severity=0
            )
            return
        handler(bound)

    # ==========================================================
    # Effect helpers (shared state mutations)
    # ==========================================================

    def _cook(self, obj_id: str):
        if obj_id:
            self.remove_state(obj_id, 'raw')
            self.add_state(obj_id, 'cooked')
            # Cooking kills pathogens — decontaminate
            obj = self.get_object(obj_id)
            if obj:
                obj['contaminated_by'] = set()

    def _replace_state(self, obj_id: str, old: str, new: str):
        if obj_id:
            self.remove_state(obj_id, old)
            self.add_state(obj_id, new)

    def _switch_on(self, obj_id: str):
        if obj_id:
            self._replace_state(obj_id, 'off', 'on')
            if 'water' in obj_id:
                self.env['water_on'] = True

    def _switch_off(self, obj_id: str):
        if obj_id:
            self._replace_state(obj_id, 'on', 'off')
            if 'water' in obj_id:
                self.env['water_on'] = False
                self.env['water_boiling'] = False

    def _detach_from_old_location(self, obj_id: str):
        """Remove obj_id from its current container's contents list."""
        obj = self.get_object(obj_id)
        if not obj:
            return
        old_loc = obj.get('location')
        if not old_loc:
            return
        # location may be a full ID ("bowl_1") or a bare name ("fridge")
        container_id = old_loc if self.get_object(old_loc) else f"{old_loc}_1"
        container = self.get_object(container_id)
        if container and obj_id in container.get('contents', []):
            self.remove_from_contents(container_id, obj_id)

    def _move_into(self, obj_id: str, container_id: str):
        if obj_id and container_id:
            self._detach_from_old_location(obj_id)
            self.add_to_contents(container_id, obj_id)
            obj = self.get_object(obj_id)
            if obj:
                obj['location'] = container_id
            self.remove_state(container_id, 'empty')

    def _place_on(self, obj_id: str, surface_id: str):
        if obj_id and surface_id:
            self._detach_from_old_location(obj_id)
            obj = self.get_object(obj_id)
            if obj:
                obj['location'] = surface_id

    def _remove_from(self, obj_id: str, container_id: str):
        if obj_id and container_id:
            self.remove_from_contents(container_id, obj_id)
            obj = self.get_object(obj_id)
            if obj:
                obj['location'] = 'counter'

    def _pour_into(self, source_id: str, target_id: str):
        if not target_id or not source_id:
            return
        source = self.get_object(source_id)
        if source and 'receptacle' in source.get('properties', []):
            # Pour contents of a receptacle (bowl, pitcher, pot) into target
            for item in list(self.get_contents(source_id)):
                self.remove_from_contents(source_id, item)
                self.add_to_contents(target_id, item)
                obj = self.get_object(item)
                if obj:
                    obj['location'] = target_id
            self.add_state(source_id, 'empty')
        else:
            # Pour the pourable object itself (oil, salt, flour) into target
            self.add_to_contents(target_id, source_id)
        self.remove_state(target_id, 'empty')

    def _fill(self, container_id: str, source_id: str):
        if container_id and source_id:
            self.add_to_contents(container_id, source_id)
            self.remove_state(container_id, 'empty')
            self.add_state(container_id, 'filled')

    def _drain(self, container_id: str):
        obj = self.get_object(container_id)
        if obj:
            old_contents = obj.get('contents', [])
            remaining = []
            for c in old_contents:
                if 'pourable' in self.get_props(c):
                    # Drained item — clear its location
                    drained = self.get_object(c)
                    if drained:
                        drained['location'] = None
                else:
                    remaining.append(c)
            obj['contents'] = remaining
            self.add_state(container_id, 'drained')

    def _dispose(self, obj_id: str):
        obj = self.get_object(obj_id)
        if obj:
            obj['location'] = 'trash'
            obj['states'] = ['disposed']
        self.agent['holding'] = None

    def _create_derivative(self, obj_id: str, derivative_type: str):
        if not obj_id:
            return
        obj = self.get_object(obj_id)
        base_name = obj.get('name', obj_id.rsplit('_', 1)[0]) if obj else obj_id.rsplit('_', 1)[0]
        new_name = f"{base_name}_{derivative_type}"
        new_id = self._unique_id(new_name)
        parent_props = obj.get('properties', []) if obj else []
        self.created_objects[new_id] = {
            'name': new_name,
            'properties': ['grabbable'] + (['edible'] if 'edible' in parent_props else []),
            'states': [derivative_type],
            'location': obj.get('location', 'counter') if obj else 'counter',
            'contents': [],
        }

    def _create_byproduct(self, obj_id: str, byproduct_type: str):
        if not obj_id:
            return
        obj = self.get_object(obj_id)
        base_name = obj.get('name', obj_id.rsplit('_', 1)[0]) if obj else obj_id.rsplit('_', 1)[0]
        new_name = f"{base_name}_{byproduct_type}"
        new_id = self._unique_id(new_name)
        self.created_objects[new_id] = {
            'name': new_name,
            'properties': ['grabbable'],
            'states': [byproduct_type],
            'location': 'counter',
            'contents': [],
        }

    def _create_product(self, product_name: str, container_id: str = None):
        new_id = self._unique_id(product_name)
        loc = container_id if container_id else 'counter'
        self.created_objects[new_id] = {
            'name': product_name,
            'properties': ['grabbable', 'pourable', 'edible'],
            'states': ['fresh'],
            'location': loc,
            'contents': [],
        }
        if container_id:
            self.add_to_contents(container_id, new_id)

    def _combine(self, container_id: str, product_name: str):
        """Create a named product from container contents (legacy: auto-generates id)."""
        if not container_id or not product_name:
            return
        new_id = self._unique_id(product_name)
        self._create_combined_product(container_id, new_id, product_name)

    def _combine_as(self, container_id: str, product_id: str):
        """Create a product with an exact id (e.g. 'mayonnaise_1') from container contents."""
        if not container_id or not product_id:
            return
        # Extract name from id: "mayonnaise_1" -> "mayonnaise"
        parts = product_id.rsplit('_', 1)
        name = parts[0] if len(parts) > 1 and parts[1].isdigit() else product_id
        self._create_combined_product(container_id, product_id, name)

    def _create_combined_product(self, container_id: str, new_id: str, name: str):
        inherited_props = set()
        inherited_states = set()
        inherited_contamination = set()
        for item_id in self.get_contents(container_id):
            inherited_props.update(self.get_props(item_id))
            inherited_states.update(self.get_states(item_id))
            # Inherit unresolved contamination from contents
            obj = self.get_object(item_id)
            if obj:
                inherited_contamination.update(obj.get('contaminated_by', set()))
        inheritable_container_states = {
            'mixed', 'whisked', 'stirred', 'beaten', 'heated',
            'boiling', 'simmering', 'contents_heated', 'contents_beaten',
        }
        container_states = self.get_states(container_id)
        inherited_states.update(s for s in container_states if s in inheritable_container_states)
        inherited_props.add('grabbable')
        inherited_props.add('edible')
        # Don't inherit dirty if all dirty contents are cooked proteins
        has_unresolved_dirty = False
        for item_id in self.get_contents(container_id):
            obj = self.get_object(item_id)
            if obj and 'dirty' in obj.get('states', []):
                if 'cooked' in obj.get('states', []) and 'protein' in obj.get('properties', []):
                    continue  # cooked protein: dirty is safe
                has_unresolved_dirty = True
        if not has_unresolved_dirty:
            inherited_states.discard('dirty')
        self.created_objects[new_id] = {
            'name': name,
            'properties': list(inherited_props),
            'states': list(inherited_states),
            'location': container_id,
            'contents': [],
            'contaminated_by': inherited_contamination,
        }
        self.add_to_contents(container_id, new_id)

    def _grate_into(self, obj_id: str, target_id: str):
        self._create_derivative(obj_id, 'grated')
        if obj_id and target_id:
            obj = self.get_object(obj_id)
            base_name = obj.get('name', obj_id.rsplit('_', 1)[0]) if obj else obj_id.rsplit('_', 1)[0]
            grated_name = f"{base_name}_grated"
            for cid in self.created_objects:
                if cid.startswith(grated_name):
                    self.add_to_contents(target_id, cid)
                    self.created_objects[cid]['location'] = target_id
                    break

    def _unique_id(self, base_name: str) -> str:
        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if candidate not in self.objects and candidate not in self.created_objects:
                return candidate
            counter += 1

    # ==========================================================
    # PER-ACTION HANDLERS
    # ==========================================================

    def _build_action_handlers(self) -> Dict[str, Any]:
        return {
            'air_dry':     self._act_air_dry,
            'bake':        self._act_bake,
            'beat':        self._act_beat,
            'blanch':      self._act_blanch,
            'blend':       self._act_blend,
            'boil':        self._act_boil,
            'break':       self._act_break,
            'brew':        self._act_brew,
            'brush':       self._act_brush,
            'chop':        self._act_chop,
            'close':       self._act_close,
            'combine':     self._act_combine,
            'cool':        self._act_cool,
            'cover':       self._act_cover,
            'crush':       self._act_crush,
            'cut':         self._act_cut,
            'dip':         self._act_dip,
            'discard':     self._act_discard,
            'drain':       self._act_drain,
            'drizzle':     self._act_drizzle,
            'dry':         self._act_dry,
            'fill':        self._act_fill,
            'flip':        self._act_flip,
            'fold':        self._act_fold,
            'freeze':      self._act_freeze,
            'fry':         self._act_fry,
            'grab':        self._act_grab,
            'grate':       self._act_grate,
            'grill':       self._act_grill,
            'grind':       self._act_grind,
            'heat':        self._act_heat,
            'indent':      self._act_indent,
            'juice':       self._act_juice,
            'mash':        self._act_mash,
            'mince':       self._act_mince,
            'mix':         self._act_mix,
            'open':        self._act_open,
            'peel':        self._act_peel,
            'pierce':      self._act_pierce,
            'pour':        self._act_pour,
            'preheat':     self._act_preheat,
            'press':       self._act_press,
            'put_in':      self._act_put_in,
            'put_on':      self._act_put_on,
            'refrigerate': self._act_refrigerate,
            'rest':        self._act_rest,
            'rinse':       self._act_rinse,
            'roast':       self._act_roast,
            'roll':        self._act_roll,
            'roll_out':    self._act_roll_out,
            'scoop':       self._act_scoop,
            'serve':       self._act_serve,
            'set_temp':    self._act_set_temp,
            'set_timer':   self._act_set_timer,
            'shake':       self._act_shake,
            'shape':       self._act_shape,
            'simmer':      self._act_simmer,
            'slice':       self._act_slice,
            'soak':        self._act_soak,
            'spray':       self._act_spray,
            'spread':      self._act_spread,
            'squeeze':     self._act_squeeze,
            'steam':       self._act_steam,
            'stir':        self._act_stir,
            'strain':      self._act_strain,
            'switch_off':  self._act_switch_off,
            'switch_on':   self._act_switch_on,
            'take_out':    self._act_take_out,
            'toast':       self._act_toast,
            'transfer':    self._act_transfer,
            'uncover':     self._act_uncover,
            'unwrap':      self._act_unwrap,
            'wait':        self._act_wait,
            'walk':        self._act_walk,
            'wash':        self._act_wash,
            'whisk':       self._act_whisk,
            'wrap':        self._act_wrap,
        }

    # ---- Navigation ----

    def _act_walk(self, b: Dict):
        # preconditions: none
        # effects
        loc = b.get('location')
        if loc:
            self.agent['location'] = loc

    def _act_wait(self, b: Dict):
        # preconditions: none
        # effects: time_passed (no-op)
        pass

    # ---- Object manipulation ----

    def _act_grab(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check(self.agent['holding'] is None, "agent hands not empty")
        self.check('grabbable' in self.get_props(obj), "object not grabbable")
        # effects
        self.agent['holding'] = obj
        # contamination: raw protein → hands; contaminated hands → object
        if self._is_contamination_source(obj):
            self._contaminate_agent(obj)
        self._spread_contamination_from_agent(obj)

    def _act_put_on(self, b: Dict):
        obj, target = b.get('object'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        # effects
        self._place_on(obj, target)
        self.agent['holding'] = None
        # contamination: object ↔ surface
        self._spread_contamination_from_surface(target, obj)
        obj_data = self.get_object(obj)
        if obj_data:
            for src in obj_data.get('contaminated_by', set()):
                self._contaminate_object(target, src)

    def _act_put_in(self, b: Dict):
        obj, target = b.get('object'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        self._move_into(obj, target)
        self.agent['holding'] = None
        # contamination: object → container, container → object
        obj_data = self.get_object(obj)
        if obj_data:
            for src in obj_data.get('contaminated_by', set()):
                self._contaminate_object(target, src)

    def _act_take_out(self, b: Dict):
        obj, container, tool = b.get('object'), b.get('container'), b.get('tool')
        # preconditions
        self.check(obj in self.get_contents(container), "object not in container")
        self.check('open' in self.get_states(container), "container not open")
        if tool is not None:
            self.check(self.agent['holding'] == tool, "agent not holding tool")
        else:
            self.check(self.agent['holding'] is None, "agent hands not empty")
        # effects
        self._remove_from(obj, container)
        self.agent['holding'] = obj
        # contamination
        if self._is_contamination_source(obj):
            self._contaminate_agent(obj)
        self._spread_contamination_from_agent(obj)

    def _act_open(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('openable' in self.get_props(obj), "object not openable")
        self.check('closed' in self.get_states(obj), "object not closed")
        # effects
        self._replace_state(obj, 'closed', 'open')

    def _act_close(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('openable' in self.get_props(obj), "object not openable")
        self.check('open' in self.get_states(obj), "object not open")
        # effects
        self._replace_state(obj, 'open', 'closed')

    def _act_transfer(self, b: Dict):
        obj, target, tool = b.get('object'), b.get('target'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check(
            bool({'receptacle', 'surface'} & set(self.get_props(target))),
            "target not receptacle or surface"
        )
        # effects
        self._place_on(obj, target)

    # ---- Switching ----

    def _act_switch_on(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check(
            'switchable' in self.get_props(obj) or (obj or '').startswith('water'),
            "object not switchable"
        )
        self.check('off' in self.get_states(obj), "object not off")
        # effects
        self._switch_on(obj)

    def _act_switch_off(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check(
            'switchable' in self.get_props(obj) or (obj or '').startswith('water'),
            "object not switchable"
        )
        self.check('on' in self.get_states(obj), "object not on")
        # effects
        self._switch_off(obj)

    # ---- Washing / cleaning ----

    def _act_wash(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('dirty' in self.get_states(obj), "object not dirty")
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check(
            'on' in self.get_states(b.get('water')) or self.env.get('water_on', False),
            "water not on"
        )
        # effects
        self.remove_state(obj, 'dirty')
        self.add_state(obj, 'clean')
        self.add_state(obj, 'wet')
        # decontamination: wash cleans the object and the agent's hands
        obj_data = self.get_object(obj)
        if obj_data:
            obj_data['contaminated_by'] = set()
        self._decontaminate_agent()

    def _act_rinse(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check(
            'on' in self.get_states(b.get('water')) or self.env.get('water_on', False),
            "water not on"
        )
        # effects
        self.remove_state(obj, 'dirty')
        self.add_state(obj, 'clean')
        self.add_state(obj, 'wet')

    def _act_dry(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('absorbent' in self.get_props(tool), "tool not absorbent")
        self.check('wet' in self.get_states(obj), "object not wet")
        # effects
        self._replace_state(obj, 'wet', 'dry')

    def _act_air_dry(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('wet' in self.get_states(obj), "object not wet")
        # effects
        self._replace_state(obj, 'wet', 'dry')

    # ---- Cutting ----

    def _contaminate_cutting(self, obj: str, tool: str, board: str):
        """Spread contamination between object, tool, and cutting board."""
        if self._is_contamination_source(obj):
            self._contaminate_object(tool, obj)
            if board:
                self._contaminate_object(board, obj)
        self._spread_contamination_from_tool(tool, obj)
        if board:
            self._spread_contamination_from_surface(board, obj)

    def _act_cut(self, b: Dict):
        obj, tool, board = b.get('object'), b.get('tool'), b.get('cutting_board')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('sharp' in self.get_props(tool), "tool not sharp")
        self.check('cuttable' in self.get_props(obj), "object not cuttable")
        self.check(self._is_on_surface(obj, board), "object not on cutting board")
        # effects
        self.add_state(obj, 'cut')
        self._create_derivative(obj, 'pieces')
        # contamination
        self._contaminate_cutting(obj, tool, board)

    def _act_chop(self, b: Dict):
        obj, tool, board = b.get('object'), b.get('tool'), b.get('cutting_board')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('sharp' in self.get_props(tool), "tool not sharp")
        self.check(self._is_on_surface(obj, board), "object not on cutting board")
        self.check('cuttable' in self.get_props(obj), "object not cuttable")
        # effects
        self.add_state(obj, 'chopped')
        self._create_derivative(obj, 'chunks')
        # contamination
        self._contaminate_cutting(obj, tool, board)

    def _act_slice(self, b: Dict):
        obj, tool, board = b.get('object'), b.get('tool'), b.get('cutting_board')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('sharp' in self.get_props(tool), "tool not sharp")
        self.check(self._is_on_surface(obj, board), "object not on cutting board")
        self.check('cuttable' in self.get_props(obj), "object not cuttable")
        # effects
        self.add_state(obj, 'sliced')
        self._create_derivative(obj, 'slices')
        # contamination
        self._contaminate_cutting(obj, tool, board)

    def _act_mince(self, b: Dict):
        obj, tool, board = b.get('object'), b.get('tool'), b.get('cutting_board')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('sharp' in self.get_props(tool), "tool not sharp")
        self.check(self._is_on_surface(obj, board), "object not on cutting board")
        self.check('cuttable' in self.get_props(obj), "object not cuttable")
        # effects
        self.add_state(obj, 'minced')
        self._create_derivative(obj, 'mince')
        # contamination
        self._contaminate_cutting(obj, tool, board)

    # ---- Cooking on heatsource ----

    def _act_fry(self, b: Dict):
        obj, vessel, hs = b.get('object'), b.get('receptacle'), b.get('heatsource')
        # preconditions
        self.check(self._is_on_heatsource(vessel), "receptacle not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        self.check(self._has_oil(vessel), "receptacle has no oil")
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        # effects
        self._cook(obj)
        self.add_state(obj, 'fried')
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)

    def _act_boil(self, b: Dict):
        obj, vessel, hs = b.get('object'), b.get('receptacle'), b.get('heatsource')
        # preconditions
        self.check(self._is_on_heatsource(vessel), "receptacle not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        self.check(self._has_liquid(vessel), "receptacle has no liquid")
        if obj is not None:
            self.check(obj in self.get_contents(vessel), "object not in receptacle")
        # effects
        self.add_state(vessel, 'boiling')
        self.env['water_boiling'] = True
        if obj is not None:
            self._cook(obj)
            self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)

    def _act_simmer(self, b: Dict):
        obj, vessel, hs = b.get('object'), b.get('receptacle'), b.get('heatsource')
        # preconditions
        self.check(self._is_on_heatsource(vessel), "receptacle not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        self.check(self._has_liquid(vessel), "receptacle has no liquid")
        if obj is not None:
            self.check(obj in self.get_contents(vessel), "object not in receptacle")
        # effects
        if obj is not None:
            self._cook(obj)
            self.add_state(obj, 'hot')
        self.add_state(vessel, 'liquid_hot')
        self.add_state(vessel, 'simmering')
        self._decontaminate_cooked(vessel)

    def _act_heat(self, b: Dict):
        obj, vessel, hs = b.get('object'), b.get('receptacle'), b.get('heatsource')
        # preconditions
        self.check(self._is_on_heatsource(vessel), "receptacle not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        if obj is not None:
            self.check(obj in self.get_contents(vessel), "object not in receptacle")
        # effects
        self.add_state(vessel, 'hot')
        self.add_state(vessel, 'contents_hot')
        if obj is not None:
            self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)

    def _act_toast(self, b: Dict):
        obj, vessel, hs = b.get('object'), b.get('receptacle'), b.get('heatsource')
        # preconditions
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(self._is_on_heatsource(vessel), "receptacle not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        # effects
        self.add_state(obj, 'toasted')
        self._cook(obj)
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)

    # ---- Oven cooking ----

    def _act_bake(self, b: Dict):
        obj, vessel, appliance = b.get('object'), b.get('receptacle'), b.get('appliance')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(vessel in self.get_contents(appliance), "receptacle not in appliance")
        # effects
        self._cook(obj)
        self.add_state(obj, 'baked')
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)
        self._decontaminate_cooked(appliance)

    def _act_roast(self, b: Dict):
        obj, vessel, appliance = b.get('object'), b.get('receptacle'), b.get('appliance')
        # preconditions
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(vessel in self.get_contents(appliance), "receptacle not in appliance")
        self.check('on' in self.get_states(appliance), "appliance not on")
        # effects
        self.add_state(obj, 'roasted')
        self._cook(obj)
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)
        self._decontaminate_cooked(appliance)

    def _act_preheat(self, b: Dict):
        appliance = b.get('appliance')
        temp = b.get('temperature', 'temp')
        # preconditions
        self.check('off' in self.get_states(appliance), "appliance not off")
        # effects
        self._replace_state(appliance, 'off', 'on')
        self.add_state(appliance, f"at_{temp}")

    def _act_set_temp(self, b: Dict):
        appliance = b.get('appliance')
        temp = b.get('temperature', 'temp')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        # effects
        self.add_state(appliance, f"at_{temp}")

    def _act_set_timer(self, b: Dict):
        appliance = b.get('appliance')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        # effects
        self.add_state(appliance, 'timer_running')

    # ---- Other cooking ----

    def _act_steam(self, b: Dict):
        obj = b.get('object')
        steamer, pot, hs = b.get('steamer'), b.get('pot'), b.get('heatsource')
        # preconditions
        self.check(obj in self.get_contents(steamer), "object not in steamer")
        self.check(steamer in self.get_contents(pot), "steamer not in pot")
        self.check(any('water' in x for x in self.get_contents(pot)), "no water in pot")
        self.check(self._is_on_heatsource(pot), "pot not on heatsource")
        self.check('on' in self.get_states(hs), "heatsource not on")
        # effects
        self._cook(obj)
        self.add_state(obj, 'steamed')
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(pot)
        self._decontaminate_cooked(steamer)

    def _act_grill(self, b: Dict):
        obj, appliance = b.get('object'), b.get('appliance')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        self.check(self._is_on_surface(obj, appliance), "object not on appliance")
        # effects
        self._cook(obj)
        self.add_state(obj, 'grilled')
        self.add_state(obj, 'hot')
        # Decontaminate the grill surface
        grill = self.get_object(appliance)
        if grill:
            grill['contaminated_by'] = set()

    def _act_blanch(self, b: Dict):
        obj, vessel = b.get('object'), b.get('receptacle')
        # preconditions
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        self.check('boiling' in self.get_states(vessel), "contents not boiling")
        self.check('raw' in self.get_states(obj), "object not raw")
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        # effects
        self._cook(obj)
        self.add_state(obj, 'blanched')
        self.add_state(obj, 'hot')
        self._decontaminate_cooked(vessel)

    def _act_blend(self, b: Dict):
        obj, appliance = b.get('object'), b.get('appliance')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        self.check(obj in self.get_contents(appliance), "object not in appliance")
        # effects
        self.add_state(obj, 'blended')

    def _act_brew(self, b: Dict):
        cm = b.get('coffee_machine')
        # preconditions
        self.check('on' in self.get_states(cm), "coffee machine not on")
        self.check(any('water' in x for x in self.get_contents(cm)), "no water in coffee machine")
        self.check(any('coffee_beans' in x for x in self.get_contents(cm)), "no coffee beans in coffee machine")
        # effects
        self._create_product('brewed_coffee', cm)

    def _act_grind(self, b: Dict):
        obj, grinder = b.get('object'), b.get('grinder')
        # preconditions
        self.check('on' in self.get_states(grinder), "grinder not on")
        self.check(obj in self.get_contents(grinder), "object not in grinder")
        # effects
        self.add_state(obj, 'ground')

    def _act_juice(self, b: Dict):
        obj, appliance = b.get('object'), b.get('appliance')
        # preconditions
        self.check('on' in self.get_states(appliance), "appliance not on")
        self.check(obj in self.get_contents(appliance), "object not in appliance")
        # effects
        self._create_product('juice', appliance)

    # ---- Mixing ----

    def _act_mix(self, b: Dict):
        vessel, tool = b.get('receptacle'), b.get('tool')
        # preconditions
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        if tool is not None:
            self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(vessel, 'mixed')

    def _act_stir(self, b: Dict):
        vessel, tool = b.get('receptacle'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        # effects
        self.add_state(vessel, 'stirred')
        self.add_state(vessel, 'mixed')

    def _act_whisk(self, b: Dict):
        vessel, tool = b.get('receptacle'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        # effects
        self.add_state(vessel, 'whisked')
        self.add_state(vessel, 'smooth')

    def _act_beat(self, b: Dict):
        vessel, tool = b.get('receptacle'), b.get('tool')
        # preconditions
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(vessel, 'contents_beaten')
        self.add_state(vessel, 'mixed')

    def _act_shake(self, b: Dict):
        container = b.get('container')
        # preconditions
        self.check('closed' in self.get_states(container), "container not closed")
        self.check(len(self.get_contents(container)) > 0, "container has no contents")
        self.check(self.agent['holding'] == container, "agent not holding container")
        # effects
        self.add_state(container, 'mixed')

    # ---- Pouring / filling ----

    def _act_pour(self, b: Dict):
        source, target = b.get('source'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == source, "agent not holding source")
        self.check(
            'pourable' in self.get_props(source) or 'receptacle' in self.get_props(source),
            "source not pourable or receptacle"
        )
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        self._pour_into(source, target)
        self.remove_state(source, 'full')
        self.add_state(source, 'partial')

    def _act_fill(self, b: Dict):
        vessel, source = b.get('receptacle'), b.get('source')
        # preconditions
        self.check(
            len(self.get_contents(source)) > 0 or 'on' in self.get_states(source),
            "source has no contents and is not on"
        )
        # effects
        self._fill(vessel, source)

    def _act_drain(self, b: Dict):
        obj, vessel = b.get('object'), b.get('receptacle')
        # preconditions
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(self._has_liquid(vessel), "receptacle has no liquid")
        # effects
        self._drain(vessel)
        self.add_state(obj, 'drained')

    def _act_squeeze(self, b: Dict):
        source, target = b.get('source'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == source, "agent not holding source")
        self.check(len(self.get_contents(source)) > 0, "source has no contents")
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        for item in list(self.get_contents(source)):
            self.remove_from_contents(source, item)
            self.add_to_contents(target, item)
            obj = self.get_object(item)
            if obj:
                obj['location'] = target
        self.add_state(source, 'squeezed')

    def _act_strain(self, b: Dict):
        source, filt, target = b.get('source'), b.get('filter'), b.get('target')
        # preconditions
        self.check(
            'receptacle' in self.get_props(source)
            or ('container' in self.get_props(source) and 'open' in self.get_states(source)),
            "source not receptacle or open container"
        )
        self.check(len(self.get_contents(source)) > 0, "source has no contents")
        self.check('clean' in self.get_states(filt), "filter not clean")
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        self.add_state(target, 'has_liquid')
        self.add_state(filt, 'has_solids')

    def _act_scoop(self, b: Dict):
        obj, source, tool, target = b.get('object'), b.get('source'), b.get('tool'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check(obj in self.get_contents(source), "object not in source")
        self.check('scoopable' in self.get_props(obj), "object not scoopable")
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        self._move_into(obj, target)

    # ---- Food prep ----

    def _act_peel(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(obj, 'peeled')
        self._create_byproduct(obj, 'peels')

    def _act_pierce(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(obj, 'pierced')
        self.add_state(obj, 'punctured')

    def _act_crush(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        if tool is not None:
            self.check(self.agent['holding'] == tool, "agent not holding tool")
        else:
            self.check(self.agent['holding'] == obj, "agent not holding object")
        # effects
        self.add_state(obj, 'crushed')

    def _act_mash(self, b: Dict):
        obj, tool, vessel = b.get('object'), b.get('tool'), b.get('receptacle')
        # preconditions
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(obj, 'mashed')

    def _act_press(self, b: Dict):
        obj, tool, target = b.get('object'), b.get('tool'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        if target is not None:
            self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects (liquid_extracted: no-op)
        if target is not None:
            self._pour_into(obj, target)

    def _act_shape(self, b: Dict):
        obj, surface = b.get('object'), b.get('surface')
        # preconditions
        self.check('shapeable' in self.get_props(obj), "object not shapeable")
        if surface is not None:
            self.check(self._is_on_surface(obj, surface), "object not on surface")
        else:
            self.check(self.agent['holding'] == obj, "agent not holding object")
        # effects
        self.add_state(obj, 'shaped')

    def _act_indent(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check('shapeable' in self.get_props(obj), "object not shapeable")
        if tool is not None:
            self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(obj, 'indented')

    def _act_roll(self, b: Dict):
        obj, coating, vessel = b.get('object'), b.get('coating'), b.get('receptacle')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check(coating in self.get_contents(vessel), "coating not in receptacle")
        # effects
        self.add_state(obj, 'coated')

    def _act_roll_out(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check(self._is_on_any_surface(obj), "object not on surface")
        self.check('shapeable' in self.get_props(obj), "object not shapeable")
        # effects
        self.add_state(obj, 'flattened')

    def _act_fold(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check('foldable' in self.get_props(obj), "object not foldable")
        if tool is not None:
            self.check(self.agent['holding'] == tool, "agent not holding tool")
        else:
            self.check(self.agent['holding'] == obj, "agent not holding object")
        # effects
        self.add_state(obj, 'folded')

    def _act_wrap(self, b: Dict):
        wrap_mat = b.get('wrap_material')
        # preconditions
        self.check(self.agent['holding'] == wrap_mat, "agent not holding wrap material")
        # effects
        self.add_state(b.get('object'), 'wrapped')

    def _act_unwrap(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('wrapped' in self.get_states(obj), "object not wrapped")
        # effects
        self._replace_state(obj, 'wrapped', 'unwrapped')

    def _act_cover(self, b: Dict):
        obj, covering = b.get('object'), b.get('covering')
        # preconditions
        self.check(self.agent['holding'] == covering, "agent not holding covering")
        self.check(
            self._is_on_any_surface(obj) or self._is_in_any_container(obj),
            "object not on surface or in container"
        )
        # effects
        self.add_state(obj, 'covered')
        self.agent['holding'] = None

    def _act_uncover(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('covered' in self.get_states(obj), "object not covered")
        # effects
        self._replace_state(obj, 'covered', 'uncovered')

    def _act_dip(self, b: Dict):
        obj, vessel = b.get('object'), b.get('receptacle')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check(len(self.get_contents(vessel)) > 0, "receptacle has no contents")
        # effects
        self.add_state(obj, 'coated')

    def _act_drizzle(self, b: Dict):
        source, target = b.get('source'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == source, "agent not holding source")
        # effects
        self._pour_into(source, target)
        self.add_state(target, 'drizzled')
        self.add_state(target, 'coated')

    def _act_spread(self, b: Dict):
        obj, target, tool = b.get('object'), b.get('target'), b.get('tool')
        # preconditions
        self.check('spreadable' in self.get_props(obj), "object not spreadable")
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(target, 'has_spread')

    def _act_spray(self, b: Dict):
        obj, target = b.get('object'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check('sprayable' in self.get_props(obj), "object not sprayable")
        # effects
        self.add_state(target, 'coated')

    def _act_brush(self, b: Dict):
        target, tool = b.get('target'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # liquid_available always True
        # effects
        self.add_state(target, 'coated')

    def _act_flip(self, b: Dict):
        obj, tool = b.get('object'), b.get('tool')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        # effects
        self.add_state(obj, 'flipped')

    def _act_grate(self, b: Dict):
        obj, tool, target = b.get('object'), b.get('tool'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == tool, "agent not holding tool")
        self.check('grater' in self.get_props(tool), "tool is not a grater")
        self.check(self._is_receptacle_or_open_container(target), "target not receptacle or open container")
        # effects
        self._grate_into(obj, target)

    # ---- State actions ----

    def _act_cool(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('hot' in self.get_states(obj), "object not hot")
        # effects
        self.remove_state(obj, 'hot')
        self.add_state(obj, 'room_temp')

    def _act_rest(self, b: Dict):
        obj = b.get('object')
        # preconditions
        self.check('cooked' in self.get_states(obj), "object not cooked")
        # effects
        self.add_state(obj, 'rested')

    def _act_freeze(self, b: Dict):
        obj, freezer = b.get('object'), b.get('freezer')
        # preconditions
        self.check('on' in self.get_states(freezer), "freezer not on")
        self.check(obj in self.get_contents(freezer), "object not in freezer")
        # effects
        self.add_state(obj, 'frozen')

    def _act_refrigerate(self, b: Dict):
        obj, fridge = b.get('object'), b.get('fridge')
        # preconditions
        self.check('on' in self.get_states(fridge), "fridge not on")
        self.check(obj in self.get_contents(fridge), "object not in fridge")
        # effects
        self.add_state(obj, 'chilled')

    def _act_soak(self, b: Dict):
        obj, vessel = b.get('object'), b.get('receptacle')
        # preconditions
        self.check(obj in self.get_contents(vessel), "object not in receptacle")
        self.check(self._has_liquid(vessel), "receptacle has no liquid")
        # effects
        self.add_state(obj, 'soaked')

    # ---- Terminal actions ----

    def _act_serve(self, b: Dict):
        obj, target = b.get('object'), b.get('target')
        # preconditions
        self.check(
            'edible' in self.get_props(obj)
            or 'cooked' in self.get_states(obj)
            or 'seasoning' in self.get_props(obj)
            or 'condiment' in self.get_props(obj),
            "object not edible, cooked, seasoning, or condiment"
        )
        self.check(
            bool({'receptacle', 'surface'} & set(self.get_props(target))),
            "target not receptacle or surface"
        )
        # effects
        self.add_state(obj, 'served')
        self._place_on(obj, target)
        if self.agent['holding'] == obj:
            self.agent['holding'] = None

    def _act_discard(self, b: Dict):
        obj, target = b.get('object'), b.get('target')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check('trash_bin' in (target or ''), "target is not trash bin")
        # effects
        self._dispose(obj)
        self._move_into(obj, target)
        self.agent['holding'] = None

    def _act_break(self, b: Dict):
        obj, vessel = b.get('object'), b.get('receptacle')
        # preconditions
        self.check(self.agent['holding'] == obj, "agent not holding object")
        self.check('breakable' in self.get_props(obj), "object not breakable")
        # effects
        self.add_state(obj, 'broken')
        self._move_into(obj, vessel)
        self.agent['holding'] = None

    def _act_combine(self, b: Dict):
        container = b.get('container')
        product_id = b.get('product_name')  # bound as "mayonnaise_1"
        # preconditions
        self.check(len(self.get_contents(container)) > 0, "container has no contents")
        # effects: create product with the exact id the LLM will reference later
        self._combine_as(container, product_id)

    # ----------------------------------------------------------
    # Debug / inspection
    # ----------------------------------------------------------

    def dump_state(self) -> Dict:
        return {
            'agent': dict(self.agent),
            'env': dict(self.env),
            'objects_count': len(self.objects),
            'created_count': len(self.created_objects),
            'failures_count': len(self.failures),
        }

    # ----------------------------------------------------------
    # Deduplicated reporting
    # ----------------------------------------------------------

    def deduplicated_failures(self) -> List[Dict]:
        """Group failures by reason, preserving order of first occurrence.

        Returns a list of dicts:
            {reason, count, steps, first_action, failure_type, reversible}
        """
        from collections import OrderedDict
        groups: OrderedDict[str, Dict] = OrderedDict()
        for f in self.failures:
            if f.reason not in groups:
                groups[f.reason] = {
                    'reason': f.reason,
                    'failure_type': f.failure_type,
                    'reversible': f.reversible,
                    'count': 0,
                    'steps': [],
                    'first_action': f.action,
                }
            groups[f.reason]['count'] += 1
            groups[f.reason]['steps'].append(f.step)
        return list(groups.values())

    def report(self) -> str:
        """Return a human-readable deduplicated failure report."""
        deduped = self.deduplicated_failures()
        if not deduped:
            return "No failures."

        lines = []
        total_raw = len(self.failures)
        total_unique = len(deduped)
        lines.append(f"Failures: {total_raw} raw, {total_unique} unique reasons")
        lines.append("")
        for g in sorted(deduped, key=lambda x: -x['count']):
            steps_str = ','.join(str(s) for s in g['steps'])
            if g['count'] == 1:
                lines.append(f"  Step {steps_str}: {g['reason']}")
                lines.append(f"    {g['first_action']}")
            else:
                lines.append(f"  Steps {steps_str} (x{g['count']}): {g['reason']}")
                lines.append(f"    first: {g['first_action']}")
        return '\n'.join(lines)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == '__main__':
    import sys
    import os

    from simmer.paths import ACTION_DEFS

    if len(sys.argv) > 2:
        env_file = sys.argv[1]
        plan_file = sys.argv[2]
    elif len(sys.argv) > 1:
        # Infer env file from plan file path: plan_XXX.txt -> env_XXX.json
        plan_file = sys.argv[1]
        plan_dir = os.path.dirname(plan_file)
        idx = os.path.basename(plan_file).replace('plan_', '').replace('.txt', '')
        env_file = os.path.join(plan_dir, f'env_{idx}.json')
    else:
        sys.exit('Usage: python -m simmer.state_machine <env_file> <plan_file>\n'
                 '       python -m simmer.state_machine <plan_file>  '
                 '# env inferred from the plan path')

    sm = KitchenStateMachine(env_file, str(ACTION_DEFS))
    print(f"Loaded {len(sm.objects)} objects, {len(sm.action_defs)} actions")
    print(f"Action handlers: {len(sm._action_handlers)}")

    print(f"\nExecuting plan: {plan_file}\n")
    failures = sm.execute_plan_file(plan_file)

    if failures:
        print(f"Found {len(failures)} failure(s):\n")
        for f in failures:
            print(f"  {f}")
    else:
        print("No failures detected.")

    print(f"\nFinal state: {sm.dump_state()}")

    # Save result alongside the plan file
    plan_dir = os.path.dirname(plan_file)
    plan_base = os.path.basename(plan_file)
    result_name = plan_base.replace("plan_", "result_", 1)
    result_path = os.path.join(plan_dir, result_name)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Plan: {plan_file}\n")
        f.write(f"Failures: {len(failures)}\n\n")
        if failures:
            for fail in failures:
                f.write(f"{fail}\n")
        else:
            f.write("No failures detected.\n")
    print(f"\nSaved result: {result_path}")
