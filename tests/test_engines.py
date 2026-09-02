from __future__ import annotations

import unittest

import chess
import numpy as np

import agent
from engine_aux import PositionHistory, position_key
from engine_core import MATE, SearchEngine
from engine_experimental import (
    ImplicitSearch,
    PuctSearch,
    SquareTokenPolicy,
    encode_square_tokens,
)
from engine_neural import FEATURE_COUNT, SparseNNUE


class CoreSearchTests(unittest.TestCase):
    def test_returns_legal_move_at_tiny_budget(self) -> None:
        board = chess.Board()
        move = SearchEngine().choose_move(board, 1_000, move_time_ms=5)
        self.assertIn(move, board.legal_moves)

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        engine = SearchEngine()
        move = engine.choose_move(board, 1_000, max_depth=3, move_time_ms=50)
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_mate_constant_dominates_evaluation(self) -> None:
        self.assertGreater(MATE, 10_000)

    def test_search_records_budget_and_completes_iteration(self) -> None:
        board = chess.Board()
        engine = SearchEngine()
        move = engine.choose_move(board, 1_000, max_depth=3, move_time_ms=50)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(engine.stats.allocated_soft_ms, 50)
        self.assertEqual(engine.stats.allocated_hard_ms, 50)
        self.assertGreaterEqual(engine.stats.completed_depth, 1)


class OptionalComponentTests(unittest.TestCase):
    def test_position_history_does_not_double_count_same_call(self) -> None:
        board = chess.Board()
        history = PositionHistory()
        self.assertEqual(history.observe(board), 1)
        self.assertEqual(history.observe(board), 1)
        self.assertEqual(history.count(board), 1)
        self.assertEqual(position_key(board), position_key(board.copy()))

    def test_incremental_nnue_matches_rebuild(self) -> None:
        hidden = 8
        features = np.arange(FEATURE_COUNT * hidden, dtype=np.int32)
        features = (features % 17 - 8).astype(np.int16).reshape(FEATURE_COUNT, hidden)
        bias = np.arange(hidden, dtype=np.int32)
        output = np.arange(1, hidden + 1, dtype=np.int16)
        nnue = SparseNNUE(features, bias, output, output_scale=32)
        board = chess.Board()
        parent = nnue.accumulator(board)
        move = chess.Move.from_uci("e2e4")
        child = nnue.apply_move(parent, board, move)
        board.push(move)
        rebuilt = nnue.accumulator(board)
        np.testing.assert_array_equal(child.activations, rebuilt.activations)
        self.assertEqual(nnue.evaluate(board, child), nnue.evaluate(board, rebuilt))


class ExperimentalTests(unittest.TestCase):
    def test_all_experimental_searches_return_legal_moves(self) -> None:
        for fen in (chess.STARTING_FEN, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"):
            board = chess.Board(fen)
            candidates = (
                PuctSearch().choose(board, 2),
                SquareTokenPolicy().choose(board),
                ImplicitSearch().choose(board, 2),
            )
            for move in candidates:
                self.assertIn(move, board.legal_moves)

    def test_en_passant_metadata_has_distinct_none_token(self) -> None:
        without_ep = chess.Board()
        with_ep = chess.Board("8/8/8/8/Pp6/8/8/4K2k b - a3 0 1")
        self.assertEqual(encode_square_tokens(without_ep)[-1], 19)
        self.assertEqual(encode_square_tokens(with_ep)[-1], 20 + chess.A3)

    def test_black_root_mcts_finds_mate_in_one(self) -> None:
        board = chess.Board("8/8/8/8/8/6k1/5q2/7K b - - 0 1")
        move = PuctSearch().choose(board, 100)
        board.push(move)
        self.assertTrue(board.is_checkmate())


class AgentIntegrationTests(unittest.TestCase):
    def test_default_agent_returns_legal_uci(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 500))
        self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
