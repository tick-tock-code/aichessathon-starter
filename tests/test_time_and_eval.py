"""Regression tests for time policy and the Numba numeric evaluation kernel."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import chess
import numpy as np

from engine_eval import (
    NUMBA_AVAILABLE,
    _packed_score,
    _python_packed_score,
    encode_white_perspective,
    evaluate_for_side_to_move,
    evaluate_white,
    evaluate_white_v2,
)
from engine_time import TimeManager


class TimeManagerTests(unittest.TestCase):
    def test_root_signal_scan_preserves_position(self) -> None:
        board = chess.Board()
        before = board.fen()
        signals = TimeManager.root_signals(board)
        self.assertEqual(board.fen(), before)
        self.assertEqual(signals.legal_moves, 20)
        self.assertFalse(signals.in_check)

    def test_tactical_position_gets_safer_selectivity(self) -> None:
        quiet = chess.Board()
        tactical = chess.Board("r3k2r/ppp2ppp/2n5/3q4/8/3Q4/PPP2PPP/R3K2R w KQkq - 0 1")
        manager = TimeManager()
        quiet_budget = manager.initial_budget(quiet, 30_000)
        tactical_budget = manager.initial_budget(tactical, 30_000)
        self.assertGreaterEqual(tactical_budget.soft_ms, quiet_budget.soft_ms)
        self.assertGreaterEqual(
            tactical_budget.selectivity.futility_margin, quiet_budget.selectivity.futility_margin
        )
        self.assertGreaterEqual(
            tactical_budget.selectivity.qdepth_limit, quiet_budget.selectivity.qdepth_limit
        )

    def test_budget_reserves_clock_and_has_ordered_deadlines(self) -> None:
        budget = TimeManager().initial_budget(chess.Board(), 1_000)
        self.assertGreaterEqual(budget.soft_ms, TimeManager.minimum_ms)
        self.assertLess(budget.hard_ms, 1_000)
        self.assertGreaterEqual(budget.hard_ms, budget.soft_ms)

    def test_stable_root_can_stop_at_soft_deadline(self) -> None:
        budget = TimeManager().initial_budget(chess.Board(), 20_000)
        self.assertTrue(
            TimeManager.should_stop_after_iteration(
                budget.soft_ms, budget, stable_iterations=2, score_gap=35, mate_detected=False
            )
        )
        self.assertFalse(
            TimeManager.should_stop_after_iteration(
                budget.soft_ms, budget, stable_iterations=0, score_gap=0, mate_detected=False
            )
        )

    def test_developer_ablation_overrides_are_bounded_and_separate(self) -> None:
        with patch.dict(
            "os.environ",
            {"CHESSATHON_TIME_SCALE": "0.5", "CHESSATHON_SELECTIVITY": "safe"},
        ):
            budget = TimeManager("balanced").initial_budget(chess.Board(), 30_000)
        self.assertEqual(TimeManager("balanced").time_scale, 1.0)
        self.assertEqual(budget.selectivity.lmr_max_reduction, 1)
        self.assertEqual(budget.selectivity.qdepth_limit, 10)

    def test_per_knob_selectivity_override_applies_after_preset(self) -> None:
        with patch.dict(
            "os.environ",
            {"CHESSATHON_LMR_MOVE_NUMBER": "4", "CHESSATHON_QDEPTH_LIMIT": "8"},
        ):
            budget = TimeManager("balanced").initial_budget(chess.Board(), 30_000)
        self.assertEqual(budget.selectivity.lmr_move_number, 4)
        self.assertEqual(budget.selectivity.qdepth_limit, 8)

    def test_balanced_quiet_default_uses_eight_ply_quiescence(self) -> None:
        budget = TimeManager("balanced").initial_budget(chess.Board(), 30_000)
        self.assertEqual(budget.selectivity.qdepth_limit, 8)

    def test_long_aggressive_profile_has_full_quiet_budget_and_qdepth_eight(self) -> None:
        quiet = chess.Board()
        long_budget = TimeManager("long_aggressive").initial_budget(quiet, 30_000)
        balanced_budget = TimeManager("balanced").initial_budget(quiet, 30_000)
        self.assertGreater(long_budget.soft_ms, balanced_budget.soft_ms)
        self.assertEqual(long_budget.selectivity.lmr_move_number, 2)
        self.assertEqual(long_budget.selectivity.futility_margin, 75)
        self.assertEqual(long_budget.selectivity.delta_margin, 85)
        self.assertEqual(long_budget.selectivity.qdepth_limit, 8)


class PackedEvaluationTests(unittest.TestCase):
    def test_strategic_evaluator_rewards_intact_king_shelter(self) -> None:
        board = chess.Board("4q1kr/5p1p/8/6p1/8/8/5PPP/4Q1KR w - - 0 1")
        self.assertGreater(evaluate_white_v2(board), evaluate_white(board))

    def test_strategic_evaluator_is_colour_symmetric(self) -> None:
        board = chess.Board("4q1kr/5p1p/8/6p1/8/8/5PPP/4Q1KR w - - 0 1")
        self.assertEqual(evaluate_white_v2(board), -evaluate_white_v2(board.mirror()))

    def test_encoder_uses_signed_int8_piece_codes(self) -> None:
        board = chess.Board("8/8/8/8/8/8/4p3/4K3 w - - 0 1")
        encoded = encode_white_perspective(board)
        self.assertEqual(encoded.dtype, np.int8)
        self.assertEqual(int(encoded[chess.E1]), chess.KING)
        self.assertEqual(int(encoded[chess.E2]), -chess.PAWN)

    def test_numba_kernel_matches_python_reference_on_legal_walk(self) -> None:
        board = chess.Board()
        for ply in range(80):
            encoded = encode_white_perspective(board)
            self.assertEqual(int(_packed_score(encoded)), _python_packed_score(encoded))
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[(ply * 7 + 3) % len(legal)])

    def test_mirror_and_colour_swap_negates_base_score(self) -> None:
        board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/8/1PNP1NP1/PBP1PPBP/R2Q1RK1 w - - 0 8")
        mirrored = board.mirror()
        self.assertEqual(evaluate_white(board), -evaluate_white(mirrored))

    def test_side_to_move_evaluation_flips_perspective(self) -> None:
        board = chess.Board("8/8/8/8/8/8/4k3/3QK3 w - - 0 1")
        white_score = evaluate_white(board)
        self.assertEqual(evaluate_for_side_to_move(board), white_score)
        board.turn = chess.BLACK
        self.assertEqual(evaluate_for_side_to_move(board), -white_score)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is optional outside the contest image")
    def test_numba_kernel_has_compiled_signature_after_warmup(self) -> None:
        encoded = np.zeros(64, dtype=np.int8)
        _packed_score(encoded)
        self.assertTrue(getattr(_packed_score, "signatures", ()))
