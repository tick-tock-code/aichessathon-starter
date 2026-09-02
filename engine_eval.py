"""Small, deterministic chess evaluation used by :mod:`engine_core`.

The evaluator deliberately has no learned weights.  It is useful as a robust baseline and
as a fall-back while a later, team-trained neural evaluator is being developed.  Its numeric
piece-square/material portion is Numba compiled during import; the public function remains
safe when Numba is not available locally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import chess
import numpy as np

_numba: Any | None
try:
    import numba as _numba

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - the contest image includes numba.
    _numba = None
    NUMBA_AVAILABLE = False


PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
PIECE_VALUES: Final[tuple[int, ...]] = (0, 100, 320, 330, 500, 900, 0)
MATERIAL_PHASE: Final[tuple[int, ...]] = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE: Final[int] = 24

# Tables are indexed with chess's square convention: a1 == 0.  They are intentionally
# compact and modest; search tactical correctness matters more than a fragile giant formula.
_PAWN_MG = (
    0, 0, 0, 0, 0, 0, 0, 0, -8, 2, 2, -12, -12, 2, 2, -8,
    -4, -2, 4, 16, 16, 4, -2, -4, 0, 4, 12, 24, 24, 12, 4, 0,
    4, 8, 18, 30, 30, 18, 8, 4, 12, 16, 22, 34, 34, 22, 16, 12,
    50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0,
)
_KNIGHT_MG = (
    -50, -35, -25, -25, -25, -25, -35, -50, -35, -15, 0, 5, 5, 0, -15, -35,
    -25, 5, 15, 18, 18, 15, 5, -25, -20, 8, 20, 25, 25, 20, 8, -20,
    -20, 5, 20, 28, 28, 20, 5, -20, -25, 0, 14, 20, 20, 14, 0, -25,
    -35, -15, 0, 3, 3, 0, -15, -35, -50, -35, -25, -25, -25, -25, -35, -50,
)
_BISHOP_MG = (
    -20, -12, -12, -12, -12, -12, -12, -20, -12, 0, 0, 2, 2, 0, 0, -12,
    -12, 3, 10, 12, 12, 10, 3, -12, -10, 8, 12, 18, 18, 12, 8, -10,
    -10, 5, 15, 18, 18, 15, 5, -10, -12, 8, 12, 12, 12, 12, 8, -12,
    -12, 2, 0, 0, 0, 0, 2, -12, -20, -12, -12, -12, -12, -12, -12, -20,
)
_ROOK_MG = (
    0, 0, 3, 8, 8, 3, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 3, 8, 8, 3, 0, 0,
)
_QUEEN_MG = (
    -12, -8, -8, -4, -4, -8, -8, -12, -8, 0, 0, 0, 0, 0, 0, -8,
    -8, 0, 4, 4, 4, 4, 0, -8, -4, 0, 4, 6, 6, 4, 0, -4,
    0, 0, 4, 6, 6, 4, 0, -4, -8, 4, 4, 4, 4, 4, 0, -8,
    -8, 0, 0, 0, 0, 0, 0, -8, -12, -8, -8, -4, -4, -8, -8, -12,
)
_KING_MG = (
    20, 28, 10, -5, -5, 10, 28, 20, 20, 20, 0, -12, -12, 0, 20, 20,
    -12, -18, -22, -30, -30, -22, -18, -12, -22, -30, -35, -45, -45, -35, -30, -22,
    -35, -45, -50, -60, -60, -50, -45, -35, -40, -50, -55, -65, -65, -55, -50, -40,
    -40, -50, -55, -65, -65, -55, -50, -40, -40, -50, -55, -65, -65, -55, -50, -40,
)
_KING_EG = (
    -45, -35, -25, -20, -20, -25, -35, -45, -25, -12, -5, 0, 0, -5, -12, -25,
    -20, -5, 8, 15, 15, 8, -5, -20, -15, 0, 15, 25, 25, 15, 0, -15,
    -15, 0, 15, 28, 28, 15, 0, -15, -20, -5, 10, 20, 20, 10, -5, -20,
    -25, -12, -5, 0, 0, -5, -12, -25, -45, -35, -25, -20, -20, -25, -35, -45,
)

_MG_TABLES = np.asarray(
    [np.zeros(64), _PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG],
    dtype=np.int16,
)
_EG_TABLES = np.asarray(
    [np.zeros(64), _PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_EG],
    dtype=np.int16,
)
_VALUES = np.asarray(PIECE_VALUES, dtype=np.int16)
_PHASE = np.asarray(MATERIAL_PHASE, dtype=np.int8)


def encode_white_perspective(board: chess.Board) -> np.ndarray:
    """Return 64 signed piece codes: white positive, black negative, empty zero."""
    encoded = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        encoded[square] = piece.piece_type if piece.color else -piece.piece_type
    return encoded


def _packed_score_impl(encoded: np.ndarray) -> int:
    """Evaluate an encoded board from White's perspective, without board-object access."""
    middle_game = 0
    end_game = 0
    phase = 0
    for square in range(64):
        code = int(encoded[square])
        if code == 0:
            continue
        piece = code if code > 0 else -code
        mirrored = square ^ 56
        table_square = square if code > 0 else mirrored
        sign = 1 if code > 0 else -1
        middle_game += sign * (int(_VALUES[piece]) + int(_MG_TABLES[piece, table_square]))
        end_game += sign * (int(_VALUES[piece]) + int(_EG_TABLES[piece, table_square]))
        phase += int(_PHASE[piece])
    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    return (middle_game * phase + end_game * (TOTAL_PHASE - phase)) // TOTAL_PHASE


_packed_score: Callable[[np.ndarray], int]
_numba_jit: Any | None = _numba
_packed_score = (
    _packed_score_impl
    if _numba_jit is None
    else _numba_jit.njit(cache=False)(_packed_score_impl)
)


def _python_packed_score(encoded: np.ndarray) -> int:
    """Equivalent un-jitted score, retained for minimal developer environments."""
    middle_game = 0
    end_game = 0
    phase = 0
    for square, raw_code in enumerate(encoded):
        code = int(raw_code)
        if code == 0:
            continue
        piece = abs(code)
        table_square = square if code > 0 else square ^ 56
        sign = 1 if code > 0 else -1
        middle_game += sign * (PIECE_VALUES[piece] + int(_MG_TABLES[piece, table_square]))
        end_game += sign * (PIECE_VALUES[piece] + int(_EG_TABLES[piece, table_square]))
        phase += MATERIAL_PHASE[piece]
    phase = min(phase, TOTAL_PHASE)
    return (middle_game * phase + end_game * (TOTAL_PHASE - phase)) // TOTAL_PHASE


def evaluate_white(board: chess.Board) -> int:
    """Return centipawns for White, including inexpensive positional terms."""
    encoded = encode_white_perspective(board)
    try:
        score = int(_packed_score(encoded)) if NUMBA_AVAILABLE else _python_packed_score(encoded)
    except Exception:  # pragma: no cover - protects local experimentation with numba.
        score = _python_packed_score(encoded)

    # Bishop pairs and passed pawns are unusually valuable simple terms at shallow depth.
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 28
        for square in board.pieces(chess.PAWN, color):
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            ahead = (
                chess.square_rank(enemy_square) > rank_index
                if color
                else chess.square_rank(enemy_square) < rank_index
                for enemy_square in enemy_pawns
                if abs(chess.square_file(enemy_square) - file_index) <= 1
            )
            if not any(ahead):
                advance = rank_index if color else 7 - rank_index
                score += sign * (8 + advance * 7)
    return score


def evaluate_for_side_to_move(board: chess.Board) -> int:
    """Return a score in centipawns for the side that has the next move."""
    score = evaluate_white(board)
    return score if board.turn == chess.WHITE else -score


# Compile the numeric path inside the competition's import-time allowance.
if NUMBA_AVAILABLE:
    _packed_score(np.zeros(64, dtype=np.int8))
