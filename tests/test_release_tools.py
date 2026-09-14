"""Offline checks for phase launch safety and evaluated-model packaging."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load_script("release_submit", "scripts/submit_recipe_phase.py")
registry = load_script("release_registry", "scripts/setup_playpen_registry.py")
context = load_script("release_context", "scripts/resolve_playpen_job_context.py")


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        shutil.copytree(ROOT / "configs", self.root / "configs")
        (self.root / "trainers").mkdir()
        for phase in json.loads((ROOT / "configs/recipe.json").read_text())["phases"].values():
            (self.root / phase["trainer"]).touch()
        self.parent = self.root / "parent with spaces"
        self.parent.mkdir()
        (self.parent / "adapter_config.json").write_text("{}")
        (self.parent / "adapter_model.safetensors").touch()
        self.lexicon = self.root / "environment/playpen/clembench/wordle/resources/target_words/en/official_recognized_words.txt"
        self.lexicon.parent.mkdir(parents=True)
        self.lexicon.write_text("crane\narise\n")
        root_patch = patch.object(submit, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_fresh_sft_cannot_inherit_old_adapter_or_data(self):
        inherited = {"PATH": "/bin", "HF_TOKEN": "test-only", "PLAYPEN_LOCAL_DATA": "/old/data",
                     "PLAYPEN_QWEN_PEFT_ADAPTER_PATH": "/old/adapter", "PLAYPEN_LR": "1",
                     "PLAYPEN_VENV_DIR": "/isolated/venv"}
        command, env = submit.prepare("A", None, inherited)
        self.assertEqual(env["PLAYPEN_LR"], "2e-4")
        self.assertNotIn("PLAYPEN_LOCAL_DATA", env)
        self.assertNotIn("PLAYPEN_QWEN_PEFT_ADAPTER_PATH", env)
        self.assertEqual(env["HF_TOKEN"], "test-only")
        self.assertEqual(env["PLAYPEN_VENV_DIR"], "/isolated/venv")
        self.assertEqual(command[-2], "Qwen3.5-2B")

    def test_phase_parent_validation(self):
        with self.assertRaises(ValueError):
            submit.prepare("A", self.parent, {})
        with self.assertRaises(ValueError):
            submit.prepare("B1", None, {})
        with self.assertRaises(ValueError):
            submit.prepare("B2", self.root / "missing", {})

    def test_continuations_use_absolute_existing_resources(self):
        for phase in ["B1", "B2", "C"]:
            with self.subTest(phase=phase):
                command, env = submit.prepare(phase, self.parent, {})
                self.assertTrue(Path(command[2]).is_file())
                self.assertEqual(env["PLAYPEN_QWEN_PEFT_ADAPTER_PATH"], str(self.parent))
                key = "PLAYPEN_VALID_WORDS_FILE" if phase == "C" else "PLAYPEN_WORDLE_VALID_WORDS_FILE"
                self.assertEqual(env[key], str(self.lexicon))
                resolved = context.resolve_trainer_path(self.root, self.root / "environment/playpen", command[2])
                self.assertTrue(Path(resolved).is_file())

    def test_missing_lexicon_stops_before_submission(self):
        self.lexicon.unlink()
        with self.assertRaisesRegex(ValueError, "lexicon"):
            submit.prepare("B2", self.parent, {})

    def test_branch_phase_saves_pairs_without_export_only_flag(self):
        _, env = submit.prepare("C", self.parent, {"PLAYPEN_DUMP_BRANCH_PAIRS": "/old/dump",
                                                   "PLAYPEN_BRANCH_PREF_DATA": "/old/pairs"})
        self.assertIn("PLAYPEN_SAVE_BRANCH_PAIRS", env)
        self.assertNotIn("PLAYPEN_DUMP_BRANCH_PAIRS", env)
        self.assertNotIn("PLAYPEN_BRANCH_PREF_DATA", env)
        self.assertEqual(env["PLAYPEN_DPO_BETA"], "0.2")
        self.assertEqual(env["PLAYPEN_GRADIENT_CHECKPOINTING"], "1")
        self.assertEqual(env["PLAYPEN_REQUIRE_DISTINCT_REPLY"], "0")

    def test_dry_run_does_not_submit_or_reveal_credentials(self):
        import contextlib
        import io
        output = io.StringIO()
        with patch.object(sys, "argv", ["submit_recipe_phase.py", "A", "--dry-run"]), \
                patch.dict("os.environ", {"HF_TOKEN": "do-not-print"}), \
                patch.object(submit.subprocess, "run", side_effect=AssertionError("submitted")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(submit.main(), 0)
        self.assertNotIn("do-not-print", output.getvalue())
        self.assertIn("recipe_env", json.loads(output.getvalue()))


class RegistryTests(unittest.TestCase):
    def test_plain_model_registry_does_not_select_adapter(self):
        entry = registry.build_qwen_entry("evaluated", "/model with spaces")
        self.assertEqual(entry["huggingface_id"], "/model with spaces")
        self.assertNotIn("peft_model", entry["model_config"])
        self.assertFalse(entry["model_config"]["chat_template_kwargs"]["enable_thinking"])


class PackagingTests(unittest.TestCase):
    def test_merge_preserves_fp32_and_both_eos_tokens(self):
        for original in [248044, [248044, 248046]]:
            with self.subTest(original=original):
                calls = []
                tokenizer = SimpleNamespace(unk_token_id=None,
                    convert_tokens_to_ids=lambda token: 248046,
                    save_pretrained=lambda output: calls.append("tokenizer_saved"))

                class FakeModel:
                    def __init__(self):
                        self.dtype = "bf16"
                        self.config = SimpleNamespace()
                        self.generation_config = SimpleNamespace(eos_token_id=original)

                    def float(self):
                        self.dtype = "fp32"
                        calls.append("upcast")
                        return self

                    def parameters(self):
                        return iter([SimpleNamespace(dtype=self.dtype)])

                    def merge_and_unload(self):
                        self.assert_upcast()
                        calls.append("merge")
                        return self

                    def assert_upcast(self):
                        if self.dtype != "fp32":
                            raise AssertionError("merge before upcast")

                    def save_pretrained(self, output, **kwargs):
                        calls.append("model_saved")

                model = FakeModel()
                modules = {"torch": SimpleNamespace(float32="fp32"),
                    "peft": SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda base, adapter: base)),
                    "transformers": SimpleNamespace(
                        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
                        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))}
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["merge", "--base", "base", "--adapter", "adapter", "--out", "new-output"]):
                    merge = load_script("release_merge", "scripts/merge_lora_fp32.py")
                    self.assertEqual(merge.main(), 0)
                self.assertEqual(model.generation_config.eos_token_id, [248044, 248046])
                self.assertEqual(model.config.torch_dtype, "fp32")
                self.assertEqual(calls, ["upcast", "merge", "model_saved", "tokenizer_saved"])


if __name__ == "__main__":
    unittest.main()
