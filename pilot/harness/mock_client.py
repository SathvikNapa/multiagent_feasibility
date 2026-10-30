"""Deterministic fake LLM for dry-running the harness without a GPU."""
from __future__ import annotations

import json
import random
import re


class MockLLMClient:
    """Mimics LLMClient.chat's signature. Returns scripted-ish JSON so the
    parsing/protocol/evaluator/metrics pipeline can be exercised end to end."""

    def __init__(self, behavior: str = "mixed", seed: int = 0):
        self.behavior = behavior
        self.rng = random.Random(seed)
        self.spec = type("S", (), {"name": "mock", "thinking_mode": False})()

    def chat(self, messages: list[dict], temperature: float, max_tokens: int, retries: int = 3):
        user = messages[-1]["content"]
        is_review = "self-review" in user or "team discussion" in user

        if "Team submissions:" in user:
            d = self._aggregator_response(user)
        elif "swap_adjacent" in user:
            d = self._adj_swap_response(user, is_review)
        elif "view.peek" in user:
            d = self._prefix_max_response(user, is_review)
        elif "registers.set" in user:
            d = self._bitvector_response(user, is_review)
        elif "entry_point" in user or "candidate" in user:
            d = self._ib_response(user, is_review)
        else:
            d = self._pairs_response(user, is_review)

        text = json.dumps(d)
        usage = {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4}
        return text, usage

    def _adj_swap_response(self, prompt: str, is_review: bool) -> dict:
        solve_code = (
            "def solve(workspace):\n"
            "    n = workspace.length\n"
            "    for i in range(n):\n"
            "        for j in range(n - 1 - i):\n"
            "            if workspace.peek(j) > workspace.peek(j + 1):\n"
            "                workspace.swap_adjacent(j)\n"
        )
        if "NO budget" in prompt or ("AT MOST" in prompt and "11" in prompt):
            return {
                "decision": "SOLVED", "p_feasible": 0.95, "p_team_success": 0.9,
                "solution_code": solve_code, "conflict_constraint_ids": [],
                "evidence_ids": ["ev1"], "requested_relaxations": [],
                "reason": "Bubble sort via adjacent swaps only removes an inversion per swap.",
            }
        # the F3 case: budget = inv - 1
        if self.rng.random() < 0.5:
            return {
                "decision": "INFEASIBLE", "p_feasible": 0.05, "p_team_success": 0.05,
                "solution_code": None, "conflict_constraint_ids": ["c2_swap_budget"],
                "evidence_ids": ["inversion_lower_bound"], "requested_relaxations": [],
                "reason": "The array has inversion count 11, below the lower bound of adjacent swaps needed; budget 10 is insufficient by the minimum bound argument.",
            }
        return {
            "decision": "SOLVED", "p_feasible": 0.6, "p_team_success": 0.4,
            "solution_code": solve_code, "conflict_constraint_ids": [],
            "evidence_ids": [], "requested_relaxations": [],
            "reason": "attempted bubble sort, hoping it fits under budget",
        }

    def _pairs_response(self, prompt: str, is_review: bool) -> dict:
        solve_code = "def solve(arr):\n    return sorted(arr)\n"
        if "preserve_pairs = []" in prompt or self.rng.random() < 0.3:
            return {
                "decision": "SOLVED", "p_feasible": 0.95, "p_team_success": 0.9,
                "solution_code": solve_code, "conflict_constraint_ids": [],
                "evidence_ids": [], "requested_relaxations": [],
                "reason": "plain ascending sort satisfies all consistent pairs",
            }
        if self.rng.random() < 0.5:
            return {
                "decision": "INFEASIBLE", "p_feasible": 0.1, "p_team_success": 0.1,
                "solution_code": None, "conflict_constraint_ids": ["c1_pair"],
                "evidence_ids": ["ascend_contradiction"], "requested_relaxations": [],
                "reason": "ascending order and one preserve_pairs constraint require opposite relative order for values 10 and 5",
            }
        return {
            "decision": "SOLVED", "p_feasible": 0.6, "p_team_success": 0.3,
            "solution_code": solve_code, "conflict_constraint_ids": [],
            "evidence_ids": [], "requested_relaxations": [],
            "reason": "plain sort, did not check pairs carefully",
        }

    def _prefix_max_response(self, prompt: str, is_review: bool) -> dict:
        solve_code = "def solve(view):\n    return max(view.peek(i) for i in range(view.visible_count))\n"
        if "GUARANTEED" in prompt or self.rng.random() < 0.3:
            return {
                "decision": "SOLVED", "p_feasible": 0.95, "p_team_success": 0.9,
                "solution_code": solve_code, "conflict_constraint_ids": [],
                "evidence_ids": [], "requested_relaxations": [],
                "reason": "max of the visible prefix, which is all we can observe",
            }
        if self.rng.random() < 0.5:
            return {
                "decision": "INFEASIBLE", "p_feasible": 0.05, "p_team_success": 0.05,
                "solution_code": None, "conflict_constraint_ids": ["c4_hidden_tail"],
                "evidence_ids": ["prefix_indistinguishability"], "requested_relaxations": [],
                "reason": "the hidden elements are not visible and could contain a larger value than any visible element",
            }
        return {
            "decision": "SOLVED", "p_feasible": 0.6, "p_team_success": 0.3,
            "solution_code": solve_code, "conflict_constraint_ids": [],
            "evidence_ids": [], "requested_relaxations": [],
            "reason": "assumed the guarantee held without checking",
        }

    def _bitvector_response(self, prompt: str, is_review: bool) -> dict:
        one_to_one_code = (
            "def solve(stream, registers):\n"
            "    for i in range(4):\n"
            "        registers.set(i, stream.next_bit())\n"
            "    return [registers.get(i) for i in range(4)]\n"
        )
        packed_code = (
            "def solve(stream, registers):\n"
            "    b0 = stream.next_bit(); b1 = stream.next_bit()\n"
            "    registers.set(0, b0 * 2 + b1)\n"
            "    b2 = stream.next_bit(); b3 = stream.next_bit()\n"
            "    registers.set(1, b2 * 2 + b3)\n"
            "    v0 = registers.get(0); v1 = registers.get(1)\n"
            "    return [v0 // 2, v0 % 2, v1 // 2, v1 % 2]\n"
        )
        if "EXACTLY 4 register" in prompt:
            return {
                "decision": "SOLVED", "p_feasible": 0.95, "p_team_success": 0.9,
                "solution_code": one_to_one_code, "conflict_constraint_ids": [],
                "evidence_ids": [], "requested_relaxations": [],
                "reason": "one register per bit, exact 1-1 mapping",
            }
        if "EXACTLY 2 register" in prompt:
            return {
                "decision": "SOLVED", "p_feasible": 0.9, "p_team_success": 0.7,
                "solution_code": packed_code, "conflict_constraint_ids": [],
                "evidence_ids": [], "requested_relaxations": [],
                "reason": "packed 2 bits per register, same total capacity",
            }
        # EXACTLY 3 registers (F3): correctly recognize the capacity shortfall
        return {
            "decision": "INFEASIBLE", "p_feasible": 0.05, "p_team_success": 0.05,
            "solution_code": None, "conflict_constraint_ids": ["c3_register_capacity"],
            "evidence_ids": ["pigeonhole"], "requested_relaxations": [],
            "reason": "3 one-bit registers give only 8 distinct states but 16 distinct 4-bit inputs are possible, a pigeonhole shortfall",
        }

    def _ib_response(self, prompt: str, is_review: bool) -> dict:
        m = re.search(r"exact signature `(\w+)`", prompt)
        entry_point = m.group(1) if m else "solve"
        code = f"def {entry_point}(*args, **kwargs):\n    return None\n"
        return {
            "decision": "SOLVED", "p_feasible": 0.7, "p_team_success": 0.5,
            "solution_code": code, "conflict_constraint_ids": [],
            "evidence_ids": [], "requested_relaxations": [],
            "reason": "attempted a direct implementation",
        }

    def _aggregator_response(self, prompt: str) -> dict:
        decisions = re.findall(r"decision=(\w+)", prompt)
        has_code = "has_code=yes" in prompt
        chosen = decisions[0] if decisions else "UNKNOWN"
        code = "def solve(*a, **k):\n    return None\n" if (chosen == "SOLVED" and has_code) else None
        return {
            "decision": chosen if (chosen != "SOLVED" or code) else "UNKNOWN",
            "p_feasible": 0.5, "p_team_success": 0.5,
            "solution_code": code, "conflict_constraint_ids": [],
            "evidence_ids": ["aggregator_pick"], "requested_relaxations": [],
            "reason": f"selected agent submission with decision={chosen}",
        }
