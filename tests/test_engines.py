from __future__ import annotations

import unittest
from threading import Event

import chess
import numpy as np

import agent
from engine_aux import (
    PgnEndgameTablebase,
    PolyglotBook,
    PonderController,
    PositionHistory,
    SyzygyTablebases,
    position_key,
)
from engine_core import MATE, RootCandidate, RootSearchResult, SearchEngine
from engine_eval import PIECE_VALUES
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

    def test_guard_predicts_when_another_depth_cannot_finish(self) -> None:
        self.assertFalse(SearchEngine._can_finish_next_iteration(900, 1_000, 100))
        self.assertTrue(SearchEngine._can_finish_next_iteration(100, 1_000, 100))

    def test_root_bound_gap_requires_a_complete_exact_best_move(self) -> None:
        best = chess.Move.from_uci("e2e4")
        rival = chess.Move.from_uci("d2d4")
        result = RootSearchResult(
            50,
            best,
            0,
            [RootCandidate(best, 50, 50, True), RootCandidate(rival, 0, 10, False)],
            True,
        )
        self.assertEqual(result.bound_gap, 40)
        self.assertEqual(result.close_challengers(35), [])

    def test_quiet_fork_cross_check_rejects_qd7(self) -> None:
        board = chess.Board(
            "2rqr1k1/1p3pb1/3pbnp1/p1pNp2p/2PnP3/P2PB1PP/1P2NPB1/2RQR1K1 b - - 4 16"
        )
        # The rated-game failure: ...Qd7 permits Nb6, attacking both queen and rook,
        # while Stockfish's ...Nd5 does not.
        risky = chess.Move.from_uci("d8d7")
        safe = chess.Move.from_uci("f6d5")
        self.assertGreaterEqual(
            SearchEngine._quiet_fork_liability(board, risky), PIECE_VALUES[chess.ROOK]
        )
        self.assertLess(SearchEngine._quiet_fork_liability(board, safe), PIECE_VALUES[chess.ROOK])

    def test_capturable_forker_is_not_treated_as_a_lost_rook(self) -> None:
        board = chess.Board(
            "r6k/1p1bn1b1/p3p2p/3pP3/P3q1P1/1P2Q2P/N1P2P2/4RRK1 w - - 7 32"
        )
        # After Qc5 the opposing queen can geometrically attack queen and rook,
        # but our queen can trade it off.  That is not an unavoidable rook loss.
        move = chess.Move.from_uci("e3c5")
        self.assertLess(
            SearchEngine._quiet_fork_liability(board, move), PIECE_VALUES[chess.ROOK]
        )


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

    def test_book_choice_is_deterministic_and_cached(self) -> None:
        class Entry:
            def __init__(self, move: chess.Move, weight: int) -> None:
                self.move = move
                self.weight = weight

        class Reader:
            def __init__(self) -> None:
                self.calls = 0

            def find_all(self, _: chess.Board) -> list[Entry]:
                self.calls += 1
                return [
                    Entry(chess.Move.from_uci("e2e4"), 10),
                    Entry(chess.Move.from_uci("d2d4"), 20),
                ]

        book = PolyglotBook("missing.bin")
        reader = Reader()
        book._reader = reader
        board = chess.Board()
        self.assertEqual(book.choose(board), chess.Move.from_uci("d2d4"))
        self.assertEqual(book.choose(board), chess.Move.from_uci("d2d4"))
        self.assertEqual(reader.calls, 1)
        self.assertEqual(book.stats.hits, 1)

    def test_tablebase_choice_is_legal_and_cached(self) -> None:
        class Tablebase:
            def __init__(self) -> None:
                self.calls = 0

            def probe_wdl(self, _: chess.Board) -> int:
                self.calls += 1
                return -2

            def probe_dtz(self, _: chess.Board) -> int:
                return 1

        tablebases = SyzygyTablebases("missing")
        fake = Tablebase()
        tablebases._tablebase = fake
        board = chess.Board("8/8/8/8/8/6K1/5Q2/7k w - - 0 1")
        move = tablebases.choose(board)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(tablebases.choose(board), move)
        self.assertEqual(fake.calls, len(list(board.legal_moves)))
        self.assertEqual(tablebases.stats.hits, 1)

    def test_downloaded_three_piece_export_produces_a_legal_move(self) -> None:
        tablebase = PgnEndgameTablebase()
        board = chess.Board("8/8/7k/8/7K/1P6/8/8 b - - 0 1")
        move = tablebase.choose(board)
        self.assertTrue(tablebase.enabled)
        self.assertIn(move, board.legal_moves)

    def test_ponder_portfolio_caches_exact_matching_reply(self) -> None:
        completed = Event()
        calls: list[str] = []

        def worker(fen: str, stop: Event) -> str | None:
            if stop.is_set():
                return None
            calls.append(fen)
            if fen == "second":
                completed.set()
            return f"move-for-{fen}"

        ponder = PonderController(enabled=True)
        self.assertTrue(ponder.start_many(("first", "second", "third"), worker, max_branches=2))
        self.assertTrue(completed.wait(timeout=1.0))
        self.assertTrue(ponder.stop_for_timed_search(join_timeout_s=0.1))
        self.assertEqual(calls[:2], ["first", "second"])
        self.assertNotIn("third", calls)
        self.assertEqual(ponder.take("second"), "move-for-second")
        self.assertEqual(ponder.stats.cache_hits, 1)

    def test_ponder_rejects_nonmatching_reply(self) -> None:
        completed = Event()

        def worker(_: str, __: Event) -> str:
            completed.set()
            return "e2e4"

        ponder = PonderController(enabled=True)
        self.assertTrue(ponder.start("predicted", worker))
        self.assertTrue(completed.wait(timeout=1.0))
        self.assertTrue(ponder.stop_for_timed_search(join_timeout_s=0.1))
        self.assertIsNone(ponder.take("actual"))
        self.assertEqual(ponder.stats.cache_misses, 1)


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
