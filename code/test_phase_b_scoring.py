#!/usr/bin/env python3
"""typicality.mean_nll under a fake model: per-token arrays, truncation flag, and byte-for-byte
agreement with the Phase A path at the default cap. Imports torch (project venv); no downloads."""
import unittest

import torch

from typicality import mean_nll


class FakeTok:
    """Whitespace tokenizer; add_special_tokens prepends a BOS id 1."""
    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **_):
        ids = [1] if add_special_tokens else []
        ids += [2 + (abs(hash(w)) % 50) for w in text.split()]
        return {"input_ids": torch.tensor([ids])}


class FakeModel:
    vocab = 60

    def __call__(self, ids):
        g = torch.Generator().manual_seed(int(ids.sum()))
        out = type("O", (), {})()
        out.logits = torch.randn(1, ids.shape[1], self.vocab, generator=g)
        return out


class MeanNllTests(unittest.TestCase):
    def setUp(self):
        self.tok, self.model = FakeTok(), FakeModel()
        self.text = " ".join(f"tok{i}" for i in range(40))

    def test_per_token_matches_mean_and_counts(self):
        nll, n, per, trunc = mean_nll(self.model, self.tok, self.text, "cpu", max_len=512,
                                      prefix="Writing prompt: x\n\nResponse:")
        self.assertEqual(n, 40)                # every item token scored under the prefix frame
        self.assertEqual(len(per), 40)
        self.assertAlmostEqual(nll, sum(per) / len(per), places=4)
        self.assertFalse(trunc)
        nll_s, n_s, per_s, _ = mean_nll(self.model, self.tok, self.text, "cpu", max_len=512)
        self.assertEqual(n_s, 40)              # standalone: BOS + 40 tokens, first never scored

    def test_truncation_is_reported_not_silent(self):
        nll, n, per, trunc = mean_nll(self.model, self.tok, self.text, "cpu", max_len=10,
                                      prefix="Writing prompt: x\n\nResponse:")
        self.assertTrue(trunc)
        self.assertEqual(n, 10)
        _, n2, _, trunc2 = mean_nll(self.model, self.tok, self.text, "cpu", max_len=10)
        self.assertTrue(trunc2)
        self.assertEqual(n2, 9)                # standalone: 10 ids incl. BOS, 9 scored positions


if __name__ == "__main__":
    unittest.main()
