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


def _strategic_score_impl(encoded: np.ndarray) -> int:
    """Return cheap structural and mobility terms from White's perspective."""
    white_pawns = np.zeros(8, dtype=np.int8)
    black_pawns = np.zeros(8, dtype=np.int8)
    white_king = -1
    black_king = -1
    phase = 0
    for square in range(64):
        code = int(encoded[square])
        if code == 0:
            continue
        piece = code if code > 0 else -code
        phase += int(_PHASE[piece])
        if piece == PAWN:
            if code > 0:
                white_pawns[square & 7] += 1
            else:
                black_pawns[square & 7] += 1
        elif piece == KING:
            if code > 0:
                white_king = square
            else:
                black_king = square

    score = 0
    for file_index in range(8):
        white_count = int(white_pawns[file_index])
        black_count = int(black_pawns[file_index])
        if white_count > 1:
            score -= 10 * (white_count - 1)
        if black_count > 1:
            score += 10 * (black_count - 1)
        white_neighbour = (file_index > 0 and white_pawns[file_index - 1] > 0) or (
            file_index < 7 and white_pawns[file_index + 1] > 0
        )
        black_neighbour = (file_index > 0 and black_pawns[file_index - 1] > 0) or (
            file_index < 7 and black_pawns[file_index + 1] > 0
        )
        if white_count and not white_neighbour:
            score -= 8 * white_count
        if black_count and not black_neighbour:
            score += 8 * black_count

    # King shelter matters while queens and major pieces remain. A pawn one rank
    # ahead is best; a pawn two ranks ahead still gives partial cover.
    if phase >= 8:
        for king_square, sign, pawn_code, direction in (
            (white_king, 1, PAWN, 1),
            (black_king, -1, -PAWN, -1),
        ):
            if king_square < 0:
                continue
            king_file = king_square & 7
            king_rank = king_square >> 3
            for file_delta in (-1, 0, 1):
                shield_file = king_file + file_delta
                if shield_file < 0 or shield_file > 7:
                    continue
                near_rank = king_rank + direction
                far_rank = king_rank + 2 * direction
                if 0 <= near_rank <= 7 and encoded[near_rank * 8 + shield_file] == pawn_code:
                    score += sign * 12
                elif 0 <= far_rank <= 7 and encoded[far_rank * 8 + shield_file] == pawn_code:
                    score += sign * 5
                else:
                    score -= sign * 10

    # Reward rooks on files not blocked by their own pawns, especially fully open files.
    for square in range(64):
        code = int(encoded[square])
        if code != ROOK and code != -ROOK:
            continue
        file_index = square & 7
        if code > 0 and white_pawns[file_index] == 0:
            score += 8 + (8 if black_pawns[file_index] == 0 else 0)
        elif code < 0 and black_pawns[file_index] == 0:
            score -= 8 + (8 if white_pawns[file_index] == 0 else 0)

    # A bounded mobility term catches trapped pieces without generating legal moves.
    knight_steps = (-17, -15, -10, -6, 6, 10, 15, 17)
    for square in range(64):
        code = int(encoded[square])
        piece = code if code > 0 else -code
        if piece < KNIGHT or piece > QUEEN:
            continue
        sign = 1 if code > 0 else -1
        mobility = 0
        if piece == KNIGHT:
            source_file = square & 7
            source_rank = square >> 3
            for step in knight_steps:
                target = square + step
                if target < 0 or target >= 64:
                    continue
                file_change = abs((target & 7) - source_file)
                rank_change = abs((target >> 3) - source_rank)
                if file_change * rank_change != 2:
                    continue
                target_code = int(encoded[target])
                if target_code == 0 or (target_code > 0) != (code > 0):
                    mobility += 1
            score += sign * 3 * mobility
            continue
        source_file = square & 7
        for step in (-9, -8, -7, -1, 1, 7, 8, 9):
            diagonal = step in (-9, -7, 7, 9)
            if piece == BISHOP and not diagonal:
                continue
            if piece == ROOK and diagonal:
                continue
            target = square + step
            previous_file = source_file
            while 0 <= target < 64 and abs((target & 7) - previous_file) <= 1:
                target_code = int(encoded[target])
                if target_code == 0:
                    mobility += 1
                else:
                    if (target_code > 0) != (code > 0):
                        mobility += 1
                    break
                previous_file = target & 7
                target += step
        weight = 2 if piece in (BISHOP, ROOK) else 1
        score += sign * weight * mobility
    return score


_packed_score: Callable[[np.ndarray], int]
_numba_jit: Any | None = _numba
_packed_score = (
    _packed_score_impl
    if _numba_jit is None
    else _numba_jit.njit(cache=False)(_packed_score_impl)
)
_strategic_score: Callable[[np.ndarray], int]
_strategic_score = (
    _strategic_score_impl
    if _numba_jit is None
    else _numba_jit.njit(cache=False)(_strategic_score_impl)
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


def _simple_board_terms(board: chess.Board) -> int:
    """Return the original bishop-pair and passed-pawn terms."""
    score = 0
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


def evaluate_white(board: chess.Board) -> int:
    """Return centipawns for White, including inexpensive positional terms."""
    encoded = encode_white_perspective(board)
    try:
        score = int(_packed_score(encoded)) if NUMBA_AVAILABLE else _python_packed_score(encoded)
    except Exception:  # pragma: no cover - protects local experimentation with numba.
        score = _python_packed_score(encoded)
    return score + _simple_board_terms(board)


def evaluate_for_side_to_move(board: chess.Board) -> int:
    """Return a score in centipawns for the side that has the next move."""
    score = evaluate_white(board)
    return score if board.turn == chess.WHITE else -score


def evaluate_white_v2(board: chess.Board) -> int:
    """Return the baseline plus experimental low-cost strategic terms."""
    encoded = encode_white_perspective(board)
    try:
        packed = int(_packed_score(encoded)) if NUMBA_AVAILABLE else _python_packed_score(encoded)
        strategic = int(_strategic_score(encoded))
    except Exception:  # pragma: no cover - protects minimal developer environments.
        packed = _python_packed_score(encoded)
        strategic = _strategic_score_impl(encoded)
    return packed + _simple_board_terms(board) + strategic


def evaluate_for_side_to_move_v2(board: chess.Board) -> int:
    """Return the experimental strategic score for the side to move."""
    score = evaluate_white_v2(board)
    return score if board.turn == chess.WHITE else -score


# Compile the numeric path inside the competition's import-time allowance.
if NUMBA_AVAILABLE:
    _packed_score(np.zeros(64, dtype=np.int8))
    _strategic_score(np.zeros(64, dtype=np.int8))
