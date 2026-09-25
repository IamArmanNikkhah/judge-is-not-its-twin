#!/usr/bin/env python3
"""No-network checks for the exact smoke settings, banking and stop receipt."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import colab_smoke as smoke


def row():
    text = "The key fit the moon."
    return {"item_id": "run__ff-01__zephyr-dpo__s411k0", "prompt_id": "ff-01", "text": text,
            "source_model": "HuggingFaceH4/zephyr-7b-beta", "lineage": "zephyr", "stage": "dpo",
            "run_id": "run", "frame": "completion", "params": {"k_index": 0,
            "max_new_tokens": 512, "temperature": 1., "top_p": .95, "seed": 411},
            "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
            "termination": {"generated_token_count": 10}}


class SmokeTests(unittest.TestCase):
    def test_bank_validation_and_duplicates(self):
        seen = set()
        smoke.validate_record(row(), "run", seen)
        self.assertEqual(len(seen), 1)
        with self.assertRaises(ValueError):
            smoke.validate_record(row(), "run", seen)

    def test_wrong_scope_rejected(self):
        for key, value in (("stage", "base"), ("run_id", "old"), ("frame", "chat"),
                           ("prompt_id", "ff-03"), ("text", "altered")):
            r = row()
            r[key] = value
            with self.assertRaises(ValueError):
                smoke.validate_record(r, "run", set())
        r = row()
        r["params"]["max_new_tokens"] = 220
        with self.assertRaises(ValueError):
            smoke.validate_record(r, "run", set())

    def test_real_smoke_records_validate_under_default_spec(self):
        path = Path(__file__).parent.parent / "data/runs/b0-smoke-20260905/outputs.jsonl"
        seen = set()
        for line in path.read_text().splitlines():
            smoke.validate_record(json.loads(line), "b0-completion-smoke-20260905", seen)
        self.assertEqual(len(seen), 4)
        self.assertEqual(smoke.generation_spec(**smoke.SMOKE_SPEC)["expected_records"], 4)

    def test_phase_b_spec_drives_validation(self):
        spec = smoke.generation_spec("unlikely", 768, 4, ["ff-01", "ff-03"])
        self.assertEqual(spec["expected_records"], 8)
        r = row()
        r["params"].update(max_new_tokens=768, k_index=3)
        r["prompt_id"], r["elicitation"] = "ff-03", "unlikely"
        r["item_id"] = "run__ff-03__zephyr-dpo__unlikely__s411k3"
        seen = set()
        smoke.validate_record(r, "run", seen, spec)
        self.assertEqual(seen, {r["item_id"]})
        # the same record fails under the ordinary-arm spec and under the smoke defaults
        with self.assertRaises(ValueError):
            smoke.validate_record(r, "run", set(), smoke.generation_spec("ordinary", 768, 4, ["ff-01", "ff-03"]))
        with self.assertRaises(ValueError):
            smoke.validate_record(r, "run", set())
        # an ordinary-arm record with a stray arm tag is refused
        o = row(); o["item_id"] = "run__ff-01__zephyr-dpo__unlikely__s411k0"
        with self.assertRaises(ValueError):
            smoke.validate_record(o, "run", set())
        # k index at or beyond k, token count above the cap
        r2 = row(); r2["params"]["k_index"] = 2
        with self.assertRaises(ValueError):
            smoke.validate_record(r2, "run", set())
        r3 = row(); r3["termination"]["generated_token_count"] = 513
        with self.assertRaises(ValueError):
            smoke.validate_record(r3, "run", set())
        for bad in (("five-at-once", 768, 4, ["ff-01"]), ("unlikely", 0, 4, ["ff-01"]),
                    ("unlikely", 768, 4, ["ff-01", "ff-01"]), ("unlikely", 768, 4, [])):
            with self.assertRaises(ValueError):
                smoke.generation_spec(*bad)

    def test_select_prompts_keeps_order_and_refuses_unknown(self):
        rows = [{"id": "ff-01", "text": "a"}, {"id": "ff-02", "text": "b"}, {"id": "ff-03", "text": "c"}]
        self.assertEqual([p["id"] for p in smoke.select_prompts(rows, ["ff-03", "ff-01"])], ["ff-03", "ff-01"])
        with self.assertRaises(ValueError):
            smoke.select_prompts(rows, ["ff-01", "ff-99"])

    def test_payload_command_carries_arm_and_cap(self):
        import colab_smoke_payload as payload
        bundle = {"model": "HuggingFaceH4/zephyr-7b-beta", "run_id": "pilot",
                  "generation": smoke.generation_spec("unlikely", 768, 4, ["ff-01"])}
        cmd = payload.build_command(bundle, "py", "/c/generate.py", "/c/prompts.jsonl", "/c/out.jsonl")
        self.assertEqual(cmd[cmd.index("--elicitation") + 1], "unlikely")
        self.assertEqual(cmd[cmd.index("--max-new-tokens") + 1], "768")
        self.assertEqual(cmd[cmd.index("--k") + 1], "4")
        self.assertNotIn("--resume-from", cmd)
        smoke_cmd = payload.build_command({**bundle, "generation": smoke.generation_spec(**smoke.SMOKE_SPEC)},
                                          "py", "s", "p", "o")
        self.assertEqual(smoke_cmd[smoke_cmd.index("--max-new-tokens") + 1], "512")
        self.assertEqual(smoke_cmd[smoke_cmd.index("--elicitation") + 1], "ordinary")

    def test_scoring_job_validation_and_command(self):
        import colab_smoke_payload as payload
        items = [{"item_id": "a", "prompt_id": "ff-01", "text": "x"}, {"item_id": "b", "prompt_id": "ff-02", "text": "y"}]
        job = smoke.scoring_job(items, "olmo2-base", "allenai/OLMo-2-1124-7B", 1024)
        self.assertEqual(job["expected_records"], 2)
        good = {"item_id": "a", "ref_tag": "olmo2-base", "ref_model": "allenai/OLMo-2-1124-7B",
                "frame": "completion-prefix", "max_len": 1024, "truncated": False,
                "mean_nll": 2.5, "n_tokens": 3, "token_nll": [1.0, 3.0, 3.5]}
        seen = set()
        smoke.validate_score_record(good, job, seen)
        self.assertEqual(seen, {"a"})
        for k, v in (("item_id", "zzz"), ("ref_tag", "mistral-base"), ("frame", "standalone"),
                     ("max_len", 512), ("truncated", True), ("mean_nll", float("nan")),
                     ("token_nll", [1.0, 3.0]), ("n_tokens", 0)):
            bad = dict(good, item_id="b"); bad[k] = v
            with self.assertRaises(ValueError, msg=k):
                smoke.validate_score_record(bad, job, set())
        with self.assertRaises(ValueError):          # duplicate
            smoke.validate_score_record(good, job, seen)
        with self.assertRaises(ValueError):          # duplicate ids in the item list
            smoke.scoring_job(items + [items[0]], "t", "m", 1024)
        bundle = {"model": "allenai/OLMo-2-1124-7B", "run_id": "typ", "job": job}
        cmd = payload.build_command(bundle, "py", "/c/typicality.py", "/c/prompts.jsonl", "/c/out.jsonl", "/c/items.jsonl")
        for flag, val in (("--model-tag", "olmo2-base"), ("--frame", "completion-prefix"), ("--max-len", "1024"), ("--items", "/c/items.jsonl")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)
        self.assertIn("--save-token-nll", cmd)
        self.assertIn("--fail-on-truncation", cmd)
        self.assertNotIn("--elicitation", cmd)

    def test_capture_only_endpoint_not_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertIsNone(smoke.capture_identity("s", out))
            (out / "sessions.json").write_text(json.dumps({"s": {"endpoint": "observed",
                                                                  "token": "PRIVATE"}}))
            self.assertEqual(smoke.capture_identity("s", out), {"session": "s", "endpoint": "observed"})
            self.assertNotIn("PRIVATE", (out / "identity.json").read_text())

    def test_zero_exit_missing_session_is_not_stop_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "not found", "")):
                result = smoke.stop("s", out)
            self.assertFalse(result["verified_absent"])

    def test_server_absence_is_stop_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "identity.json").write_text(json.dumps({"session": "s", "endpoint": "observed"}))
            with patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "terminated", "")), \
                 patch.object(smoke, "endpoint_check", return_value={"target_absent": True, "endpoints": []}) as check:
                result = smoke.stop("s", out)
            check.assert_called_once_with("observed")
            self.assertTrue(result["verified_absent"])


if __name__ == "__main__":
    unittest.main()


class SweepTests(unittest.TestCase):
    def test_sweep_specs_validate_per_dose(self):
        import colab_smoke_payload as payload
        specs = [smoke.generation_spec("decoding", 1024, 2, ["ff-01"], t, 0.95, run_id=f"sw-t{t}") for t in (1.3, 1.6)]
        self.assertEqual([s["run_id"] for s in specs], ["sw-t1.3", "sw-t1.6"])
        r = row()
        r.update(run_id="sw-t1.6", elicitation="decoding", item_id="sw-t1.6__ff-01__zephyr-dpo__decoding__s411k1")
        r["params"].update(max_new_tokens=1024, temperature=1.6, k_index=1)
        seen = set()
        smoke.validate_record(r, "sw", seen, specs[1])
        self.assertEqual(seen, {r["item_id"]})
        with self.assertRaises(ValueError):          # the same record against the other dose
            smoke.validate_record(r, "sw", set(), specs[0])
        with self.assertRaises(ValueError):          # temperature drift inside a dose
            bad = row(); bad.update(run_id="sw-t1.6", elicitation="decoding", item_id=r["item_id"])
            bad["params"].update(max_new_tokens=1024, temperature=1.0, k_index=1)
            smoke.validate_record(bad, "sw", set(), specs[1])
        for bad in ((0.0, 0.95), (3.5, 0.95), (1.0, 0.0), (1.0, 1.5)):
            with self.assertRaises(ValueError):
                smoke.generation_spec("decoding", 1024, 2, ["ff-01"], *bad)
        cmd = payload.build_command({"model": "m", "run_id": "sw", "generation": specs[0]}, "py", "s", "p", "o")
        self.assertEqual(cmd[cmd.index("--temperature") + 1], "1.3")
        self.assertEqual(cmd[cmd.index("--top-p") + 1], "0.95")
        self.assertEqual(cmd[cmd.index("--run-id") + 1], "sw-t1.3")
        self.assertEqual(cmd[cmd.index("--elicitation") + 1], "decoding")
        # the smoke defaults still build the frozen command
        d = payload.build_command({"model": "m", "run_id": "smk", "generation": smoke.generation_spec(**smoke.SMOKE_SPEC)}, "py", "s", "p", "o")
        self.assertEqual((d[d.index("--temperature") + 1], d[d.index("--top-p") + 1], d[d.index("--run-id") + 1]), ("1.0", "0.95", "smk"))


class TopUpTests(unittest.TestCase):
    def test_k_start_slots(self):
        import colab_smoke_payload as payload
        spec = smoke.generation_spec("decoding", 1024, 1, ["ff-01"], 1.3, 0.95, run_id="sw-t1.3", k_start=2)
        r = row(); r.update(run_id="sw-t1.3", elicitation="decoding", item_id="sw-t1.3__ff-01__zephyr-dpo__decoding__s411k2")
        r["params"].update(max_new_tokens=1024, temperature=1.3, k_index=2)
        smoke.validate_record(r, "x", set(), spec)
        for bad_index in (0, 1, 3):                  # outside the top-up window
            b = row(); b.update(run_id="sw-t1.3", elicitation="decoding", item_id=f"sw-t1.3__ff-01__zephyr-dpo__decoding__s411k{bad_index}")
            b["params"].update(max_new_tokens=1024, temperature=1.3, k_index=bad_index)
            with self.assertRaises(ValueError):
                smoke.validate_record(b, "x", set(), spec)
        cmd = payload.build_command({"model": "m", "run_id": "sw", "generation": spec}, "py", "s", "p", "o")
        self.assertEqual(cmd[cmd.index("--k-start") + 1], "2")
        self.assertEqual(cmd[cmd.index("--k") + 1], "1")
        with self.assertRaises(ValueError):
            smoke.generation_spec("decoding", 1024, 1, ["ff-01"], k_start=-1)


class SeedTests(unittest.TestCase):
    def test_seed_travels_and_is_checked(self):
        import colab_smoke_payload as payload
        spec = smoke.generation_spec("decoding", 1024, 1, ["ff-01"], 1.3, 0.95, run_id="sw-t1.3", k_start=3, seed=413)
        r = row(); r.update(run_id="sw-t1.3", elicitation="decoding", item_id="sw-t1.3__ff-01__zephyr-dpo__decoding__s413k3")
        r["params"].update(max_new_tokens=1024, temperature=1.3, k_index=3, seed=413)
        smoke.validate_record(r, "x", set(), spec)
        stale = row(); stale.update(run_id="sw-t1.3", elicitation="decoding", item_id="sw-t1.3__ff-01__zephyr-dpo__decoding__s411k3")
        stale["params"].update(max_new_tokens=1024, temperature=1.3, k_index=3, seed=411)
        with self.assertRaises(ValueError):          # a record made under the old seed is refused
            smoke.validate_record(stale, "x", set(), spec)
        cmd = payload.build_command({"model": "m", "run_id": "sw", "generation": spec}, "py", "s", "p", "o")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "413")
        self.assertEqual(payload.build_command({"model": "m", "run_id": "smk", "generation": smoke.generation_spec(**smoke.SMOKE_SPEC)}, "py", "s", "p", "o")[-5], "411")


class JudgeJobTests(unittest.TestCase):
    """Grammatical-better rerun (2026-09-25): judge job reproduces the Phase A gate settings."""
    def job(self):
        pairs = [{"prompt_id": "ff-01", "item_x": "a", "item_y": "b"}, {"prompt_id": "ff-01", "item_x": "a", "item_y": "c"}]
        return smoke.judge_job(pairs, "m", "olmo2", "sft", "better-grammatical", 1024)

    def rec(self, **kw):
        r = {"item_x": "a", "item_y": "b", "judge_model": "m", "judge_lineage": "olmo2", "judge_stage": "sft",
             "criterion": "better-grammatical", "max_len": 1024, "frame": "completion",
             "p_x": .7, "p_x_first_order": .8, "p_y_first_order": .4}
        r.update(kw)
        return r

    def test_accepts_and_rejects(self):
        job, seen = self.job(), set()
        smoke.validate_judge_record(self.rec(), job, seen)
        for bad in ({}, {"criterion": "better"}, {"item_y": "z"}, {"p_x": 1.5}, {"frame": "chat"}, {"max_len": 2048}):
            with self.assertRaises(ValueError):
                smoke.validate_judge_record(self.rec(**bad), job, set() if bad else seen)
        self.assertEqual(job["expected_records"], 2)

    def test_payload_command_matches_phase_a_gate(self):
        import colab_smoke_payload as payload
        cmd = payload.build_command({"model": "m", "job": self.job()}, "py", "judge.py", "p.jsonl", "o.jsonl",
                                    "i.jsonl", "pairs.jsonl")
        self.assertEqual(cmd[cmd.index("--criterion") + 1], "better-grammatical")
        self.assertEqual(cmd[cmd.index("--max-len") + 1], "1024")
        self.assertNotIn("--fail-on-truncation", cmd)
        self.assertNotIn("--use-chat-template", cmd)
        self.assertEqual(cmd[cmd.index("--pairs-file") + 1], "pairs.jsonl")


class RecordKeyTests(unittest.TestCase):
    def test_both_row_shapes(self):
        self.assertEqual(smoke.record_key({"item_id": "s1"}), "s1")
        self.assertEqual(smoke.record_key({"item_x": "a", "item_y": "b"}), "a|b")
