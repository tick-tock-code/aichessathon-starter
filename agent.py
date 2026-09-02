"""AI Chessathon entry point and engine composition root.

The stable competition path is the independently implemented alpha-beta engine.
Optional shipped data and experimental searches are wired behind explicit modes so
they can be benchmarked without destabilising the default agent.
"""

from __future__ import annotations

import os
from threading import Event
from typing import Final

import chess

from engine_aux import PolyglotBook, PonderController, PositionHistory, SyzygyTablebases
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
HISTORY = PositionHistory()
PONDER = PonderController(enabled=os.environ.get("CHESSATHON_PONDER") == "1")
TIME_MANAGER = TimeManager(os.environ.get("CHESSATHON_TIME_PROFILE", "balanced").strip().lower())


def _nnue_side_to_move(board: chess.Board) -> int:
    score = NNUE.evaluate(board)
    return score if board.turn == chess.WHITE else -score


SEARCH = SearchEngine(
    evaluator=_nnue_side_to_move if NNUE.enabled else None,
    policy=POLICY.score_moves if POLICY.enabled else None,
    policy_max_ply=0,
    time_manager=TIME_MANAGER,
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
    """Predict one opponent reply and search our response in the background."""
    if not PONDER.enabled:
        return
    predicted = board.copy(stack=False)
    predicted.push(our_move)
    if predicted.is_game_over(claim_draw=True):
        return
    opponent_move = TOKEN_POLICY.choose(predicted)
    predicted.push(opponent_move)
    if not predicted.is_game_over(claim_draw=True):
        PONDER.start(predicted.fen(), _ponder_worker)


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
    if ponder_stopped:
        pondered = PONDER.take(board.fen())
        if isinstance(pondered, str):
            try:
                candidate = chess.Move.from_uci(pondered)
                move = candidate if candidate in legal_moves else None
            except chess.InvalidMoveError:
                move = None
    if move is None:
        move = TABLEBASES.choose(board)
    if move is None:
        move = BOOK.choose(board)
    if move is None:
        move = _experimental_move(board, time_left_ms)

    # Every backend is designed to return legal moves. Retain a final defensive
    # barrier because one illegal response loses the game immediately.
    if move not in legal_moves:
        move = SEARCH.choose_move(board, time_left_ms)
    _start_pondering(board, move)
    return move.uci()
