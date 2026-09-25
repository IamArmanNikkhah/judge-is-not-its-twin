#!/usr/bin/env python3
"""Offline regression tests. No torch, model downloads, remote calls or judge data."""
import ast
from pathlib import Path
import random
import unittest

from phase_b_pairs import prepare


def fixture():
    items = [{"item_id": f"item-{i}", "prompt_id": "p", "source_model": "writer",
              "lineage": "family", "stage": "sft", "text": chr(65+i) * (100+i)}
             for i in range(5)]
    scores = [{"item_id": r["item_id"], "ref_tag": "heldout", "ref_model": "stranger",
               "frame": "completion-prefix", "mean_nll": 1+i/10,
               "n_tokens": 20, "n_chars": len(r["text"])} for i, r in enumerate(items)]
    return items, {"heldout": scores}, {"family": "heldout"}


class PairTests(unittest.TestCase):
    def test_generation_stop_metadata(self):
        tree = ast.parse(Path(__file__).with_name("generate.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generation_stop_metadata")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "generate.py", "exec"), ns)
        stop = ns["generation_stop_metadata"]
        self.assertEqual(stop([10, 11], 2, 2)["stop_reason"], "token_cap")
        self.assertEqual(stop([10, 2], 2, 9)["stop_reason"], "eos")
        both = stop([10, 3], [2, 3], 2)
        self.assertTrue(both["token_cap_reached"])
        self.assertTrue(both["ended_with_eos"])
        self.assertEqual(both["stop_reason"], "eos")
        self.assertEqual(stop([], None, 2)["stop_reason"], "other")
        self.assertIsNone(stop([], None, 2)["last_generated_token_id"])

    def test_disjoint_and_deterministic(self):
        items, scores, refs = fixture()
        pairs, report = prepare(items, scores, refs)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(report["unmatched_ids"], ["item-4"])
        self.assertEqual(len({p[k] for p in pairs for k in ("item_x", "item_y")}), 4)
        random.Random(42).shuffle(items)
        scores["heldout"].reverse()
        self.assertEqual(prepare(items, scores, refs), (pairs, report))

    def test_no_crossing_group_boundaries(self):
        for field in ("prompt_id", "stage", "source_model"):
            items, scores, refs = fixture()
            items[0][field] = "different"
            pairs, _ = prepare(items, scores, refs)
            bank = {x["item_id"]: x for x in items}
            for pair in pairs:
                for key in ("prompt_id", "stage", "lineage", "source_model"):
                    self.assertEqual(bank[pair["item_x"]][key], bank[pair["item_y"]][key])

    def test_duplicate_ids_rejected(self):
        items, scores, refs = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare(items + [items[0]], scores, refs)
        scores["heldout"].append(scores["heldout"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare(items, scores, refs)

    def test_missing_and_extra_scores_rejected(self):
        for extra in (False, True):
            items, scores, refs = fixture()
            if extra:
                row = dict(scores["heldout"][0], item_id="extra")
                scores["heldout"].append(row)
            else:
                scores["heldout"].pop()
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                prepare(items, scores, refs)

    def test_bad_scores_and_frames_rejected(self):
        for field, value in [("mean_nll", float("nan")), ("mean_nll", float("inf")),
                             ("mean_nll", True), ("frame", "standalone"),
                             ("ref_tag", "wrong"), ("n_chars", 99), ("n_tokens", 0)]:
            items, scores, refs = fixture()
            scores["heldout"][0][field] = value
            with self.assertRaises(ValueError):
                prepare(items, scores, refs)

    def test_reference_not_writer(self):
        items, scores, refs = fixture()
        for s in scores["heldout"]:
            s["ref_model"] = "writer"
        with self.assertRaisesRegex(ValueError, "own held-out"):
            prepare(items, scores, refs)

    def test_scores_cannot_change_pair_membership(self):
        items, scores, refs = fixture()
        first, _ = prepare(items, scores, refs)
        for i, s in enumerate(scores["heldout"]):
            s["mean_nll"] = 100 - i * 7
        second, _ = prepare(items, scores, refs)
        keys = ("item_x", "item_y", "b0_pair_id")
        self.assertEqual([[p[k] for k in keys] for p in first], [[p[k] for k in keys] for p in second])

    def test_identical_and_length_rejections_reported(self):
        items, scores, refs = fixture()
        pairs, report = prepare(items, scores, refs, max_length_ratio=1)
        self.assertEqual(pairs, [])
        self.assertEqual(len(report["length_rejected_pairs"]), 2)
        items[1]["text"] = items[0]["text"]
        scores["heldout"][1]["n_chars"] = len(items[1]["text"])
        _, report = prepare(items, scores, refs)
        self.assertEqual(len(report["identical_text_pairs_skipped"]), 1)

    def test_invalid_bounds_and_empty_bank(self):
        for bound in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare(*fixture(), max_length_ratio=bound)
        with self.assertRaisesRegex(ValueError, "empty bank"):
            prepare([], {}, {})

    def test_actual_judge_loader_compatibility(self):
        # Execute only the existing pure loader: importing judge.py would load torch.
        import json
        import tempfile
        tree = ast.parse(Path(__file__).with_name("judge.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_pairs_file")
        ns = {"json": json}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "judge.py", "exec"), ns)
        items, scores, refs = fixture()
        pairs, _ = prepare(items, scores, refs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
            loaded = ns["load_pairs_file"](path, items)
        self.assertEqual(len(loaded), len(pairs))
        self.assertTrue(all(x["source_model"] == y["source_model"] for _, x, y, _ in loaded))
        self.assertTrue(all(k.startswith("b0_") for _, _, _, meta in loaded for k in meta))

    def test_review_blinding_and_score_independence(self):
        from phase_b_review import build_review
        items, scores, refs = fixture()
        pairs, _ = prepare(items, scores, refs)
        prompts = [{"id": "p", "text": "Write a story."}]
        review, key = build_review(items, pairs, prompts, "family", "sft")
        self.assertEqual(len(key), 1)
        for secret in ("heldout", "writer", "b0_", "mean_nll", "item-0"):
            self.assertNotIn(secret, review)
        for p in pairs:
            p["b0_delta_nll"] = 999
        review2, _ = build_review(items, pairs, prompts, "family", "sft")
        self.assertEqual(review, review2)
        pairs[0]["prompt_id"] = "wrong"
        with self.assertRaises(ValueError):
            build_review(items, pairs, prompts, "family", "sft")


if __name__ == "__main__":
    unittest.main()
