from __future__ import annotations

import json
import unittest
from pathlib import Path

from .config import load_config
from .runner import condition_sets, resolve_plan
from .tasks import load_tasks


class OrchestratorOfflineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.config = load_config(self.root / "edu_experiment/settings.example.json")
        self.tasks = load_tasks(self.config.edu_sample / "tasks/tasks.jsonl")

    def test_full_conditions_use_only_required_surfaces(self) -> None:
        self.assertEqual(len(condition_sets({"graph", "db", "kb"})), 7)
        self.assertEqual(condition_sets({"graph", "db"})[0][1], frozenset({"graph", "db"}))

    def test_example_plan_supports_three_modes(self) -> None:
        plan = json.loads((self.root / "edu_experiment/ablation_plan.example.json").read_text())
        resolved = {item.task["task_id"]: item for item in resolve_plan(self.tasks, plan)}
        self.assertEqual(resolved["edu_001"].mode, "full")
        self.assertEqual(len(resolved["edu_001"].conditions), 7)
        self.assertEqual(resolved["edu_002"].conditions[0].active_surfaces, frozenset({"graph", "kb"}))
        self.assertEqual(resolved["edu_020"].mode, "none")


if __name__ == "__main__":
    unittest.main()
