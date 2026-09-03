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
PONDER = PonderController(enabled=os.environ.get("CHESSATHON_PONDER") == "1")
TIME_MANAGER = TimeManager(os.environ.get("CHESSATHON_TIME_PROFILE", "balanced").strip().lower())


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


PONDER_BRANCHES: Final = _bounded_env_int("CHESSATHON_PONDER_BRANCHES", 3, 1, 4)


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
    }
    print("CHESSATHON_METRIC " + json.dumps(payload, separators=(",", ":"), sort_keys=True))


SEARCH = SearchEngine(
    evaluator=_nnue_side_to_move if NNUE.enabled else None,
    policy=POLICY.score_moves if POLICY.enabled else None,
    policy_max_ply=0,
    time_manager=TIME_MANAGER,
    verify_root_scores=os.environ.get("CHESSATHON_VERIFY_ROOT") == "1",
)
TOKEN_POLICY = SquareTokenPolicy()
EXPERTS = PhaseExpertRouter()
MCTS = PuctSearch(policy=TOKEN_POLICY.scores, value=EXPERTS.evaluate)
IMPLICIT = ImplicitSearch(policy=TOKEN_POLICY, router=EXPERTS)


def _ponder_worker(predicted_fen: str, stop: Event) -> str | None:
    """Search one predicted reply on isolated state during the opponent's clock."""
    if stop.is_set():
        return None
    board = chess.Board(predicted_fen)
    if board.is_game_over(claim_draw=True):
        return None
    search = SearchEngine(table_limit=30_000)
    move = search.choose_move(board, 5_000, move_time_ms=750, cancel=stop.is_set)
    return None if stop.is_set() else move.uci()


def _start_pondering(board: chess.Board, our_move: chess.Move) -> None:
    """Prepare responses to a small policy-ranked portfolio of opponent replies."""
    if not PONDER.enabled:
        return
    after_our_move = board.copy(stack=False)
    after_our_move.push(our_move)
    if after_our_move.is_game_over(claim_draw=True):
        return
    replies = list(after_our_move.legal_moves)
    try:
        scores = TOKEN_POLICY.scores(after_our_move, replies)
        replies.sort(key=lambda move: (scores.get(move, float("-inf")), move.uci()), reverse=True)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        replies.sort(key=chess.Move.uci)
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

    # Pondering is opt-in and currently disabled. This remains the foreground
    # safety barrier if a future experiment enables it.
    ponder_stopped = PONDER.stop_for_timed_search()
    HISTORY.observe(board)

    move: chess.Move | None = None
    source = "search"
    if ponder_stopped:
        pondered = PONDER.take(board.fen())
        if isinstance(pondered, str):
            try:
                candidate = chess.Move.from_uci(pondered)
                move = candidate if candidate in legal_moves else None
                if move is not None:
                    source = "ponder"
            except chess.InvalidMoveError:
                move = None
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
