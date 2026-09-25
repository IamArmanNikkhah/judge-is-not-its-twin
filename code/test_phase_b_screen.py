#!/usr/bin/env python3
"""Offline tests for the Phase B screen, pairing, manipulation check and the pure helpers
added to generate.py / judge.py. No torch, no models, no remote calls. The four real
09-05 smoke records are the positive control for the screen."""
import ast
import json
from pathlib import Path
import unittest

from phase_b_screen import (screen_record, screen_bank, exclusion_report, pair_arms,
                            token_budget_check, MIN_WORDS, MAX_RATIO)
from phase_b_manip_check import paired_deltas, summarize

HERE = Path(__file__).parent
SMOKE = HERE.parent / "data" / "runs" / "b0-smoke-20260905" / "outputs.jsonl"


def pure_functions(filename, *names):
    """exec only the named top-level defs/assigns of a torch-importing script."""
    tree = ast.parse((HERE / filename).read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), filename, "exec"), ns)
    return ns


def story(words, arm="ordinary", pid="p1", ident=None, cap=768, n_tok=None, stop="eos", tail="."):
    text = " ".join(f"w{i}" for i in range(words)) + tail
    n = n_tok if n_tok is not None else words + 1
    return {"item_id": ident or f"{arm}-{pid}-{words}", "prompt_id": pid, "text": text,
            "elicitation": arm, "params": {"max_new_tokens": cap},
            "termination": {"stop_reason": stop, "generated_token_count": n}}


class ScreenTests(unittest.TestCase):
    def test_real_smoke_records(self):
        recs = [json.loads(l) for l in SMOKE.read_text().splitlines() if l.strip()]
        self.assertEqual(len(recs), 4)
        verdict = {r["item_id"].split("__")[-1] + "/" + r["prompt_id"]: screen_record(r) for r in recs}
        # ff-01/k0 hit the 512 cap and ends with "and": two reasons, nothing else fails
        self.assertEqual(sorted(verdict["s411k0/ff-01"]), ["no_eos", "no_terminal_punctuation"])
        self.assertEqual(verdict["s411k1/ff-01"], [])
        self.assertEqual(verdict["s411k0/ff-02"], [])   # EOS at 510 of 512: before the cap, kept
        self.assertEqual(verdict["s411k1/ff-02"], [])

    def test_each_reason_fires_alone(self):
        self.assertEqual(screen_record(story(120)), [])
        self.assertEqual(screen_record(story(120, stop="token_cap")), ["no_eos"])
        self.assertEqual(screen_record(story(120, n_tok=768)), ["eos_at_cap"])
        self.assertEqual(screen_record(story(120, tail=" and")), ["no_terminal_punctuation"])
        self.assertEqual(screen_record(story(120, tail='.”')), [])
        leak = story(120); leak["text"] += "\n\nWriting prompt: another"
        self.assertEqual(screen_record(leak), ["no_terminal_punctuation", "leaked_frame"])
        ph = story(120); ph["text"] = ph["text"][:-1] + " [insert name]."
        self.assertEqual(screen_record(ph), ["placeholder"])
        for slot in ("[Name]", "{name}", "[YOUR NAME]"):
            ph = story(120); ph["text"] = ph["text"][:-1] + f" {slot}."
            self.assertEqual(screen_record(ph), ["placeholder"], slot)
        # an in-story bracket is not a template slot (pilot ff-03: a letter signed "[Name withheld]")
        ok = story(120); ok["text"] = ok["text"][:-1] + "\n\nSincerely,\n\n[Name withheld]\n\nShe folded the letter."
        self.assertEqual(screen_record(ok), [])
        self.assertEqual(screen_record(story(MIN_WORDS - 1)), ["below_floor"])
        self.assertEqual(screen_record(story(MIN_WORDS)), [])

    def test_identical_text_within_arm_and_prompt(self):
        a, b = story(120, ident="a"), story(120, ident="b")
        c = story(120, arm="unlikely", ident="c")          # same text, other arm: allowed
        kept, excluded = screen_bank([a, b, c])
        self.assertEqual([r["item_id"] for r in kept], ["a", "c"])
        self.assertEqual(excluded[0]["reasons"], ["identical_text"])

    def test_tripwire_per_arm_and_gap(self):
        recs = [story(120, ident=f"o{i}") for i in range(20)] + \
               [story(120, arm="unlikely", ident=f"u{i}") for i in range(20)]
        rep = exclusion_report(recs, [])
        self.assertFalse(rep["tripwire"]["tripped"])
        # 3/20 = 15% in one arm trips the arm threshold
        exc = [{"item_id": f"u{i}", "arm": "unlikely", "prompt_id": "p1", "reasons": ["no_eos"], "words": 0}
               for i in range(3)]
        rep = exclusion_report(recs, exc)
        self.assertTrue(rep["tripwire"]["tripped"])
        self.assertEqual(rep["per_arm"]["unlikely"]["reasons"], {"no_eos": 3})
        # 2/20 = 10% in one arm, 0 in the other: under the arm bar, over the 5-point gap
        rep = exclusion_report(recs, exc[:2])
        self.assertTrue(rep["tripwire"]["tripped"])
        self.assertAlmostEqual(rep["arm_gap"], 0.10)
        # 1/20 each: clear
        exc2 = [dict(exc[0]), dict(exc[0], item_id="o0", arm="ordinary")]
        self.assertFalse(exclusion_report(recs, exc2)["tripwire"]["tripped"])


class PairTests(unittest.TestCase):
    def test_closest_greedy_and_ratio(self):
        kept = [story(200, ident="o200"), story(300, ident="o300"),
                story(210, arm="unlikely", ident="u210"), story(400, arm="unlikely", ident="u400")]
        pairs, rejected, unmatched = pair_arms(kept)
        got = {tuple(sorted((p["item_x"], p["item_y"]))) for p in pairs}
        self.assertEqual(got, {("o200", "u210")})           # 300 vs 400 is 1.33 > 1.25
        self.assertEqual(rejected[0]["control"], "o300")
        self.assertEqual(unmatched, [])                      # rejected items are used, not unmatched
        pairs2, _, _ = pair_arms(kept, max_ratio=1.5)
        self.assertEqual(len(pairs2), 2)

    def test_orientation_is_stable_and_spans_arms(self):
        kept = [story(200, ident="o"), story(205, arm="unlikely", ident="u")]
        p1, _, _ = pair_arms(kept)
        p2, _, _ = pair_arms(list(reversed(kept)))
        self.assertEqual(p1, p2)
        self.assertEqual({p1[0]["pb_arm_x"], p1[0]["pb_arm_y"]}, {"ordinary", "unlikely"})
        self.assertAlmostEqual(abs(p1[0]["pb_log_word_ratio"]), abs(__import__("math").log(200 / 205)))

    def test_no_cross_prompt_pairs_and_unmatched(self):
        kept = [story(200, ident="o1", pid="p1"), story(200, arm="unlikely", ident="u2", pid="p2")]
        pairs, rejected, unmatched = pair_arms(kept)
        self.assertEqual(pairs, [])
        self.assertEqual(sorted(unmatched), ["o1", "u2"])

    def test_token_budget_check(self):
        kept = [story(300, ident="o"), story(310, arm="unlikely", ident="u")]
        pairs, _, _ = pair_arms(kept)
        items = {r["item_id"]: r for r in kept}
        tok = lambda s: s.split()
        frame = "Prompt: {prompt}\n\nResponse A: {a}\n\nResponse B: {b}\n\nThe more {criterion} response is Response"
        rep = token_budget_check(pairs, items, {"p1": "write"}, tok, 2048, frame, 1024, "Writing prompt: write Response:")
        self.assertTrue(rep["judge_fits"] and rep["scorer_fits"])
        self.assertEqual(rep["longest_item"], 310)
        rep = token_budget_check(pairs, items, {"p1": "write"}, tok, 600, frame, 1024, "x")
        self.assertFalse(rep["judge_fits"])


class ManipCheckTests(unittest.TestCase):
    def _pairs_scores(self, deltas):
        pairs, scores = [], {}
        for i, d in enumerate(deltas):
            o, u = f"o{i}", f"u{i}"
            pairs.append({"prompt_id": "p", "item_x": o, "item_y": u, "pb_arm_x": "ordinary",
                          "pb_arm_y": "unlikely", "pb_pair_id": str(i)})
            scores[o] = {"item_id": o, "frame": "completion-prefix", "mean_nll": 2.0, "truncated": False}
            scores[u] = {"item_id": u, "frame": "completion-prefix", "mean_nll": 2.0 + d, "truncated": False}
        return pairs, scores

    def test_direction_and_interval(self):
        pairs, scores = self._pairs_scores([0.3, 0.2, 0.4, 0.25, 0.35, 0.3, 0.2, 0.4])
        res = summarize(paired_deltas(pairs, scores))
        self.assertTrue(res["manipulation_detected"])
        self.assertEqual(res["frac_treatment_less_typical"], 1.0)
        pairs, scores = self._pairs_scores([0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
        self.assertFalse(summarize(paired_deltas(pairs, scores))["manipulation_detected"])

    def test_refuses_truncated_or_wrong_frame(self):
        pairs, scores = self._pairs_scores([0.3])
        scores["u0"]["truncated"] = True
        with self.assertRaises(ValueError):
            paired_deltas(pairs, scores)
        scores["u0"]["truncated"] = False
        scores["o0"]["frame"] = "standalone"
        with self.assertRaises(ValueError):
            paired_deltas(pairs, scores)


class HelperTests(unittest.TestCase):
    def test_generate_frames_and_ids(self):
        ns = pure_functions("generate.py", "COMPLETION_FRAME", "UNLIKELY_FRAME", "ELICITATIONS",
                            "build_frames", "item_id_for")
        seen, ruler = ns["build_frames"]("Write a story.", "ordinary")
        self.assertEqual(seen, ruler)
        self.assertEqual(ruler, "Writing prompt: Write a story.\n\nResponse:")
        seen_u, ruler_u = ns["build_frames"]("Write a story.", "unlikely")
        self.assertEqual(ruler_u, ruler)                        # common evaluation frame
        self.assertIn("under 10 percent", seen_u)
        self.assertTrue(seen_u.startswith("Writing prompt: Write a story.\n\n"))
        self.assertTrue(seen_u.endswith("\n\nResponse:"))
        with self.assertRaises(ValueError):
            ns["build_frames"]("x", "five-at-once")
        # Phase A id format is byte-identical for the ordinary arm; the other arm carries its tag
        self.assertEqual(ns["item_id_for"]("b0-completion-smoke-20260905", "ff-01", "zephyr", "dpo", "ordinary", 411, 0),
                         "b0-completion-smoke-20260905__ff-01__zephyr-dpo__s411k0")
        self.assertEqual(ns["item_id_for"](None, "ff-01", "zephyr", "dpo", "unlikely", 411, 3),
                         "ff-01__zephyr-dpo__unlikely__s411k3")

    def test_judge_fit_max_len(self):
        ns = pure_functions("judge.py", "fit_max_len")
        ia, ib, trunc, longest = ns["fit_max_len"]([1, 2, 3], [1, 2, 4], 3)
        self.assertEqual((ia, ib, trunc, longest), ([1, 2, 3], [1, 2, 4], False, 3))
        ia, ib, trunc, longest = ns["fit_max_len"]([1, 2, 3, 4], [1, 2, 5], 3)
        self.assertEqual((ia, ib, trunc, longest), ([2, 3, 4], [2, 5], True, 4))


if __name__ == "__main__":
    unittest.main()


class DecodingArmTests(unittest.TestCase):
    def test_decoding_frames_and_ids(self):
        ns = pure_functions("generate.py", "COMPLETION_FRAME", "UNLIKELY_FRAME", "ELICITATIONS", "build_frames", "item_id_for")
        seen, ruler = ns["build_frames"]("Write a story.", "decoding")
        self.assertEqual(seen, ruler)                    # same ordinary frame; the dose is in params
        self.assertEqual(ns["item_id_for"]("sw-t1.3", "ff-01", "zephyr", "dpo", "decoding", 411, 0),
                         "sw-t1.3__ff-01__zephyr-dpo__decoding__s411k0")

    def test_screen_pairs_named_treatment_arm(self):
        kept = [story(200, ident="o"), story(205, arm="decoding", ident="d"), story(210, arm="unlikely", ident="u")]
        pairs, _, unmatched = pair_arms([k for k in kept if k["elicitation"] != "unlikely"], arms=("ordinary", "decoding"))
        self.assertEqual({pairs[0]["pb_arm_x"], pairs[0]["pb_arm_y"]}, {"ordinary", "decoding"})
        with self.assertRaises(ValueError):              # a foreign arm in the input is refused
            pair_arms(kept, arms=("ordinary", "decoding"))
        rep = exclusion_report(kept[:2], [], arms=("ordinary", "decoding"))
        self.assertEqual(set(rep["per_arm"]), {"ordinary", "decoding"})
        # manip check accepts any single treatment arm
        p = {"prompt_id": "p", "item_x": "o", "item_y": "d", "pb_arm_x": "ordinary", "pb_arm_y": "decoding", "pb_pair_id": "1"}
        sc = {"o": {"item_id": "o", "frame": "completion-prefix", "mean_nll": 1.0, "truncated": False},
              "d": {"item_id": "d", "frame": "completion-prefix", "mean_nll": 1.4, "truncated": False}}
        self.assertAlmostEqual(paired_deltas([p], sc)[0], 0.4)
