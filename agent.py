"""AI Chessathon entry point and engine composition root.

The stable competition path is the independently implemented alpha-beta engine.
Optional shipped data and experimental searches are wired behind explicit modes so
they can be benchmarked without destabilising the default agent.
"""

from __future__ import annotations

import json
import os
from threading import Event
from typing import Final

import chess

from engine_aux import (
    PgnEndgameTablebase,
    PolyglotBook,
    PonderController,
    PositionHistory,
    SyzygyTablebases,
)
from engine_core import SearchEngine
from engine_experimental import ImplicitSearch, PhaseExpertRouter, PuctSearch, SquareTokenPolicy
from engine_neural import ONNXPolicy, SparseNNUE
from engine_time import TimeManager

SUPPORTED_MODES: Final = frozenset({"alphabeta", "mcts", "dag", "policy", "implicit", "moe"})
requested_mode = os.environ.get("CHESSATHON_ENGINE", "alphabeta").strip().lower()
ENGINE_MODE: Final = requested_mode if requested_mode in SUPPORTED_MODES else "alphabeta"

# Import happens once per game inside the separate 60-second initialisation budget.
NNUE = SparseNNUE.from_shipped_weights()
POLICY = ONNXPolicy()
BOOK = PolyglotBook()
TABLEBASES = SyzygyTablebases()
PGN_TABLEBASES = PgnEndgameTablebase()
HISTORY = PositionHistory()
PONDER = PonderController(enabled=os.environ.get("CHESSATHON_PONDER", "1") != "0")
TIME_MANAGER = TimeManager(
    os.environ.get("CHESSATHON_TIME_PROFILE", "long_aggressive").strip().lower()
)


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


PONDER_BRANCHES: Final = _bounded_env_int("CHESSATHON_PONDER_BRANCHES", 3, 1, 4)
PONDER_SLICE_MS: Final = _bounded_env_int("CHESSATHON_PONDER_SLICE_MS", 125, 40, 300)
PONDER_SEARCHES: dict[str, SearchEngine] = {}


def _nnue_side_to_move(board: chess.Board) -> int:
    score = NNUE.evaluate(board)
    return score if board.turn == chess.WHITE else -score


def _emit_metrics(time_left_ms: int, source: str, board: chess.Board, move: chess.Move) -> None:
    """Write developer telemetry to stderr via the local harness, when requested."""
    if os.environ.get("CHESSATHON_METRICS") != "1":
        return
    stats = SEARCH.stats
    payload = {
        "profile": TIME_MANAGER.profile,
        "time_scale": TIME_MANAGER.time_scale,
        "selectivity": TIME_MANAGER.selectivity_override,
        "time_manager": SEARCH.time_manager_mode,
        "source": source,
        "fen": board.fen(),
        "move": move.uci(),
        "time_left_ms": time_left_ms,
        "elapsed_ms": stats.elapsed_ms if source == "search" else 0,
        "allocated_soft_ms": stats.allocated_soft_ms if source == "search" else 0,
        "allocated_hard_ms": stats.allocated_hard_ms if source == "search" else 0,
        "depth": stats.completed_depth if source == "search" else 0,
        "nodes": stats.nodes if source == "search" else 0,
        "qnodes": stats.qnodes if source == "search" else 0,
        "tt_hits": stats.tt_hits if source == "search" else 0,
        "cutoffs": stats.cutoffs if source == "search" else 0,
        "best_score": stats.root_best_score if source == "search" else 0,
        "second_score": stats.root_second_score if source == "search" else 0,
        "stable_iterations": stats.stable_iterations if source == "search" else 0,
        "best_move_changes": stats.best_move_changes if source == "search" else 0,
        "root_urgency": stats.root_urgency if source == "search" else 0,
        "root_legal_moves": stats.root_legal_moves if source == "search" else 0,
        "root_checking_moves": stats.root_checking_moves if source == "search" else 0,
        "root_capturing_moves": stats.root_capturing_moves if source == "search" else 0,
        "root_scores_verified": stats.root_scores_verified if source == "search" else False,
        "iterations_started": stats.iterations_started if source == "search" else 0,
        "iterations_completed": stats.iterations_completed if source == "search" else 0,
        "aborted_depth": stats.aborted_depth if source == "search" else 0,
        "last_iteration_ms": stats.last_iteration_ms if source == "search" else 0,
        "next_iteration_skipped": stats.next_iteration_skipped if source == "search" else 0,
        "root_bound_gap": stats.root_bound_gap if source == "search" else 0,
        "challenger_verifications": stats.challenger_verifications if source == "search" else 0,
        "stop_reason": stats.stop_reason if source == "search" else "",
    }
    print("CHESSATHON_METRIC " + json.dumps(payload, separators=(",", ":"), sort_keys=True))


SEARCH = SearchEngine(
    evaluator=_nnue_side_to_move if NNUE.enabled else None,
    policy=POLICY.score_moves if POLICY.enabled else None,
    policy_max_ply=0,
    time_manager=TIME_MANAGER,
    verify_root_scores=os.environ.get("CHESSATHON_VERIFY_ROOT") == "1",
    time_manager_mode=os.environ.get("CHESSATHON_TIME_MANAGER", "guarded"),
)
TOKEN_POLICY = SquareTokenPolicy()
EXPERTS = PhaseExpertRouter()
MCTS = PuctSearch(policy=TOKEN_POLICY.scores, value=EXPERTS.evaluate)
IMPLICIT = ImplicitSearch(policy=TOKEN_POLICY, router=EXPERTS)


def _ponder_worker(predicted_fen: str, stop: Event) -> SearchEngine | None:
    """Advance one cancellable slice of a predicted reply's isolated search."""
    if stop.is_set():
        return None
    board = chess.Board(predicted_fen)
    if board.is_game_over(claim_draw=True):
        return None
    search = PONDER_SEARCHES.get(predicted_fen)
    if search is None:
        search = SearchEngine(
            table_limit=30_000,
            evaluator=_nnue_side_to_move if NNUE.enabled else None,
            time_manager=TIME_MANAGER,
            time_manager_mode=SEARCH.time_manager_mode,
        )
        PONDER_SEARCHES[predicted_fen] = search
    search.choose_move(board, 5_000, move_time_ms=PONDER_SLICE_MS, cancel=stop.is_set)
    return None if stop.is_set() else search


def _reply_priority(
    board: chess.Board, move: chess.Move, recapture_square: chess.Square
) -> tuple[int, int, int, str]:
    """Rank plausible opponent replies without relying on an untrained policy."""
    is_capture = board.is_capture(move)
    victim = board.piece_at(move.to_square)
    victim_value = 0 if victim is None else victim.piece_type
    centrality = (
        7 - abs(3 - chess.square_file(move.to_square)) - abs(3 - chess.square_rank(move.to_square))
    )
    return (
        int(move.promotion is not None) * 10_000 + int(board.gives_check(move)) * 5_000,
        int(is_capture) * 1_000 + int(move.to_square == recapture_square) * 500 + victim_value,
        centrality,
        move.uci(),
    )


def _start_pondering(board: chess.Board, our_move: chess.Move) -> None:
    """Continuously prepare a forcing, non-policy portfolio during opponent time."""
    if not PONDER.enabled:
        return
    after_our_move = board.copy(stack=False)
    after_our_move.push(our_move)
    if after_our_move.is_game_over(claim_draw=True):
        return
    replies = list(after_our_move.legal_moves)
    replies.sort(
        key=lambda move: _reply_priority(after_our_move, move, our_move.to_square), reverse=True
    )
    PONDER_SEARCHES.clear()
    predicted_fens: list[str] = []
    for opponent_move in replies[:PONDER_BRANCHES]:
        predicted = after_our_move.copy(stack=False)
        predicted.push(opponent_move)
        if not predicted.is_game_over(claim_draw=True):
            predicted_fens.append(predicted.fen())
    PONDER.start_many(predicted_fens, _ponder_worker, max_branches=PONDER_BRANCHES)


def _move_budget_ms(time_left_ms: int) -> int:
    """Conservative wall-time allocation shared by experimental modes."""
    remaining = max(1, time_left_ms - 10)
    return min(remaining, max(25, min(4_000, time_left_ms // 24 + 120)))


def _experimental_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    budget = _move_budget_ms(time_left_ms)
    if ENGINE_MODE in {"mcts", "dag", "moe"}:
        return MCTS.choose(board, budget)
    if ENGINE_MODE == "policy":
        return TOKEN_POLICY.choose(board, budget)
    if ENGINE_MODE == "implicit":
        return IMPLICIT.choose(board, budget)
    return SEARCH.choose_move(board, time_left_ms)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for ``fen`` within the supplied game clock."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # The referee never requests a move from a finished game, but failing
        # explicitly makes malformed external calls easier to diagnose.
        raise ValueError("get_move called for a terminal position")

    # Stop the sole background worker before foreground search touches any
    # transferred table entries. An immediate opponent reply safely falls back
    # to a normal search if the worker has not yielded yet.
    ponder_stopped = PONDER.stop_for_timed_search()
    HISTORY.observe(board)

    move: chess.Move | None = None
    source = "search"
    if ponder_stopped:
        pondered = PONDER.take(board.fen())
        if isinstance(pondered, SearchEngine):
            SEARCH.absorb_pondered_search(pondered)
            PONDER_SEARCHES.clear()
    if move is None:
        move = TABLEBASES.choose(board)
        if move is not None:
            source = "syzygy"
    if move is None:
        move = PGN_TABLEBASES.choose(board)
        if move is not None:
            source = "endgame_pgn"
    if move is None:
        move = BOOK.choose(board)
        if move is not None:
            source = "book"
    if move is None:
        move = _experimental_move(board, time_left_ms)

    # Every backend is designed to return legal moves. Retain a final defensive
    # barrier because one illegal response loses the game immediately.
    if move not in legal_moves:
        move = SEARCH.choose_move(board, time_left_ms)
        source = "search"
    _start_pondering(board, move)
    _emit_metrics(time_left_ms, source, board, move)
    return move.uci()
