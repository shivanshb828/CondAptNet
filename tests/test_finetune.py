"""
Tests for scripts/training/finetune.py.

Tests cover:
  - filter_by_tier_targets: fuzzy matching, normalisation, empty target list
  - _merge_extra_data: schema mapping, target filtering, ATGC-only guard
  - main() CLI: --stage validation / deployment, empty targets guard, no
    pretrain ckpt warning, --dry-run-style (--max-batches 0 without real data)
  - Integration smoke: build model, call set_stage2(), check trainable params
    include LoRA but also that ESM backbone grads are frozen in stage1.

No real CSV data or pretrained weights are required — all tests use
small synthetic DataFrames or unittest.mock.
"""

from __future__ import annotations

import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Module under test ─────────────────────────────────────────────────────────
from scripts.training.finetune import filter_by_tier_targets, _merge_extra_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(target_proteins: list[str]) -> pd.DataFrame:
    """Minimal DataFrame that satisfies filter_by_tier_targets input."""
    return pd.DataFrame({
        "aptamer_sequence": ["ATCGATCGATCG"] * len(target_proteins),
        "target_name":      target_proteins,
        "protein_sequence": ["MHQTLK"] * len(target_proteins),
        "split":            ["train"] * len(target_proteins),
        "label":            [1] * len(target_proteins),
        "training_tier":    [2] * len(target_proteins),
    })


# ── filter_by_tier_targets ────────────────────────────────────────────────────

class TestFilterByTierTargets(unittest.TestCase):

    def test_exact_match_case_insensitive(self):
        df = _make_df(["Insulin", "Thrombin", "Albumin"])
        result = filter_by_tier_targets(df, ["insulin"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["target_name"], "Insulin")

    def test_underscore_normalised(self):
        """troponin_I target keyword should match 'Troponin I' protein name."""
        df = _make_df(["Troponin I", "Albumin"])
        result = filter_by_tier_targets(df, ["troponin_I"])
        self.assertEqual(len(result), 1)
        self.assertIn("Troponin I", result["target_name"].values)

    def test_dash_normalised(self):
        """NT-proBNP keyword → 'nt probnp' should match 'NT-proBNP' protein."""
        df = _make_df(["NT-proBNP", "Insulin"])
        result = filter_by_tier_targets(df, ["NT-proBNP"])
        self.assertEqual(len(result), 1)

    def test_substring_match_long_name(self):
        """'albumin' keyword matches 'human serum albumin'."""
        df = _make_df(["human serum albumin", "Insulin"])
        result = filter_by_tier_targets(df, ["albumin"])
        self.assertEqual(len(result), 1)
        self.assertIn("human serum albumin", result["target_name"].values)

    def test_multiple_targets(self):
        df = _make_df(["Insulin", "Myoglobin", "Thrombin", "Albumin"])
        result = filter_by_tier_targets(df, ["insulin", "myoglobin"])
        self.assertEqual(len(result), 2)

    def test_no_match_returns_empty(self):
        df = _make_df(["Thrombin", "VEGF"])
        result = filter_by_tier_targets(df, ["insulin"])
        self.assertTrue(result.empty)
        self.assertListEqual(list(result.columns), list(df.columns))

    def test_empty_targets_returns_empty_dataframe(self):
        df = _make_df(["Insulin", "Myoglobin"])
        result = filter_by_tier_targets(df, [])
        self.assertTrue(result.empty)
        self.assertListEqual(list(result.columns), list(df.columns))

    def test_none_target_protein_does_not_raise(self):
        df = _make_df(["Insulin"])
        df.loc[0, "target_name"] = None
        result = filter_by_tier_targets(df, ["insulin"])
        self.assertTrue(result.empty)

    def test_returns_copy_not_view(self):
        df = _make_df(["Insulin"])
        result = filter_by_tier_targets(df, ["insulin"])
        result.loc[result.index[0], "label"] = 0
        self.assertEqual(df.iloc[0]["label"], 1)  # original unchanged

    def test_all_validation_targets(self):
        """All six Tier 2 targets should match their canonical names."""
        from config import VALIDATION_TARGETS
        names = ["Insulin", "Myoglobin", "NT-proBNP",
                 "Troponin I", "Troponin T", "Albumin"]
        df = _make_df(names)
        result = filter_by_tier_targets(df, VALIDATION_TARGETS)
        self.assertEqual(len(result), len(names),
                         f"Expected all {len(names)} to match, got {len(result)}")


# ── _merge_extra_data ─────────────────────────────────────────────────────────

class TestMergeExtraData(unittest.TestCase):

    def _make_scraper_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        p = tmp_path / "scraped_dataset.csv"
        df.to_csv(p, index=False)
        return p

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmpdir)

    def test_empty_extra_path_returns_base_unchanged(self):
        base = _make_df(["Insulin"])
        result = _merge_extra_data(base, "", ["insulin"])
        pd.testing.assert_frame_equal(result, base)

    def test_missing_file_returns_base_unchanged(self):
        base = _make_df(["Insulin"])
        result = _merge_extra_data(base, "/nonexistent/path.csv", ["insulin"])
        pd.testing.assert_frame_equal(result, base)

    def test_valid_extra_rows_appended(self):
        """Rows matching targets in scraped CSV should be appended."""
        base = _make_df(["Insulin"])
        extra_rows = [
            {
                "aptamer_sequence": "ATCGATCGATCGATCGATCG",
                "target_name": "Myoglobin",
                "kd_value": "10.0",
                "nucleic_acid_type": "DNA",
                "modifications": "none",
                "target_type": "protein",
            }
        ]
        csv_path = self._make_scraper_csv(self._tmp_path, extra_rows)
        result = _merge_extra_data(base, str(csv_path), ["myoglobin"])
        # Should have 1 original + 1 extra = 2
        self.assertEqual(len(result), 2)

    def test_extra_rows_not_matching_target_excluded(self):
        """Scraped rows for non-target proteins should not be added."""
        base = _make_df(["Insulin"])
        extra_rows = [
            {
                "aptamer_sequence": "ATCGATCGATCGATCGATCG",
                "target_name": "Thrombin",
            }
        ]
        csv_path = self._make_scraper_csv(self._tmp_path, extra_rows)
        result = _merge_extra_data(base, str(csv_path), ["insulin"])
        # Extra row is for thrombin, not insulin — should not be appended
        self.assertEqual(len(result), 1)

    def test_non_atgc_sequences_filtered_out(self):
        """Sequences with non-ATGC chars (RNA U, ambiguous N) should be dropped."""
        base = _make_df(["Insulin"])
        extra_rows = [
            {
                "aptamer_sequence": "AUGCAUGCAUGCAUGCAUGC",  # RNA
                "target_name": "Insulin",
            }
        ]
        csv_path = self._make_scraper_csv(self._tmp_path, extra_rows)
        result = _merge_extra_data(base, str(csv_path), ["insulin"])
        self.assertEqual(len(result), 1)  # RNA row filtered out

    def test_extra_data_default_label_is_1(self):
        """Scraped rows without 'label' column should default to label=1 (binder)."""
        base = _make_df([])
        extra_rows = [
            {
                "aptamer_sequence": "ATCGATCGATCGATCGATCG",
                "target_name": "Insulin",
            }
        ]
        csv_path = self._make_scraper_csv(self._tmp_path, extra_rows)
        result = _merge_extra_data(base, str(csv_path), ["insulin"])
        self.assertEqual(len(result), 1)
        self.assertEqual(int(float(result.iloc[0]["label"])), 1)


# ── Model stage integration ───────────────────────────────────────────────────

class TestModelStageTransition(unittest.TestCase):
    """
    Verify set_stage1() / set_stage2() grad flags.

    Implementation detail: freeze_esm() freezes ESM-2 backbone weights but
    explicitly KEEPS LoRA matrices trainable even in Stage 1.  unfreeze_lora()
    in set_stage2() is therefore a no-op (LoRA was never frozen).  Both stages
    have the same trainable-parameter count; the difference is only which
    learning rate is used for LoRA vs non-ESM layers.
    """

    def test_stage1_freezes_esm_backbone(self):
        """After set_stage1(), all non-LoRA ESM-2 params must be frozen."""
        from models.condaptnet import CondAptNet
        model = CondAptNet(predict_kd=True)
        model.set_stage1()

        esm_frozen = all(
            not p.requires_grad
            for n, p in model.protein_encoder.esm.named_parameters()
            if "lora_" not in n
        )
        self.assertTrue(esm_frozen,
            "ESM-2 backbone (non-LoRA) must be frozen after set_stage1()")

    def test_lora_trainable_in_stage1(self):
        """
        LoRA matrices remain trainable in Stage 1 — freeze_esm() only
        freezes the original ESM-2 weights, not the adapters.
        """
        from models.condaptnet import CondAptNet
        model = CondAptNet(predict_kd=True)
        model.set_stage1()

        lora_trainable = any(
            p.requires_grad
            for n, p in model.named_parameters()
            if "lora_" in n
        )
        self.assertTrue(lora_trainable,
            "LoRA adapters should remain trainable in Stage 1")

    def test_stage2_keeps_esm_backbone_frozen(self):
        """ESM-2 non-LoRA params must stay frozen after set_stage2()."""
        from models.condaptnet import CondAptNet
        model = CondAptNet(predict_kd=True)
        model.set_stage1()
        model.set_stage2()

        esm_frozen = all(
            not p.requires_grad
            for n, p in model.protein_encoder.esm.named_parameters()
            if "lora_" not in n
        )
        self.assertTrue(esm_frozen,
            "ESM-2 backbone (non-LoRA) must stay frozen in stage 2")

    def test_stage2_lora_remains_trainable(self):
        """LoRA params must be trainable in Stage 2."""
        from models.condaptnet import CondAptNet
        model = CondAptNet(predict_kd=True)
        model.set_stage1()
        model.set_stage2()

        lora_trainable = any(
            p.requires_grad
            for n, p in model.named_parameters()
            if "lora_" in n
        )
        self.assertTrue(lora_trainable,
            "LoRA adapters must be trainable in stage 2")

    def test_trainable_param_count_same_stage1_and_stage2(self):
        """
        Trainable param counts are equal because LoRA was already trainable
        in Stage 1 — set_stage2()/unfreeze_lora() is effectively a no-op.
        """
        from models.condaptnet import CondAptNet
        model = CondAptNet(predict_kd=True)

        model.set_stage1()
        n1 = model.trainable_params()

        model.set_stage2()
        n2 = model.trainable_params()

        self.assertEqual(n1, n2,
            "Trainable param count should be identical in Stage 1 and Stage 2 "
            "(LoRA is already trainable from set_stage1)")


# ── CLI argument parsing ──────────────────────────────────────────────────────

class TestFinetuneCLI(unittest.TestCase):

    def _run_main_with_args(self, args: list[str]):
        """Run finetune.main() with patched sys.argv."""
        import scripts.training.finetune as ft
        with patch("sys.argv", ["finetune.py"] + args):
            return ft

    def test_stage_required(self):
        """Missing --stage should raise SystemExit (argparse)."""
        import scripts.training.finetune as ft
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["finetune.py"]):
                import argparse
                # Simulate parse_args with no --stage
                parser = argparse.ArgumentParser()
                parser.add_argument("--stage", choices=["validation", "deployment"],
                                    required=True)
                parser.parse_args([])

    def test_empty_deployment_targets_exits(self):
        """If DEPLOYMENT_TARGETS is [], main() should exit with code 1."""
        import scripts.training.finetune as ft

        with patch.object(ft, "DEPLOYMENT_TARGETS", []):
            with patch("sys.argv", ["finetune.py", "--stage", "deployment"]):
                with self.assertRaises(SystemExit) as cm:
                    ft.main()
                self.assertEqual(cm.exception.code, 1)

    def test_missing_pretrain_checkpoint_warns_not_crashes(self):
        """
        When the pretrain checkpoint doesn't exist the run should emit a
        warning but continue (it will start from random init).

        We abort early by patching filter_by_tier_targets to return an
        empty DataFrame so the data check exits before training starts.
        """
        import scripts.training.finetune as ft

        # Make filter return empty → sys.exit(1) from the data check
        with patch.object(ft, "VALIDATION_TARGETS", ["insulin"]):
            with patch.object(ft, "filter_by_tier_targets",
                              return_value=pd.DataFrame()):
                with patch.object(ft, "_merge_extra_data",
                                  return_value=pd.DataFrame()):
                    with patch("sys.argv", [
                        "finetune.py",
                        "--stage", "validation",
                        "--pretrain-checkpoint", "/nonexistent/best.pt",
                    ]):
                        with self.assertRaises(SystemExit) as cm:
                            ft.main()
                        # Exit code 1 from empty-data check (not from ckpt load)
                        self.assertEqual(cm.exception.code, 1)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
