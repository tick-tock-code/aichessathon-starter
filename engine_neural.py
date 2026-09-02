"""Optional learned evaluation and policy ordering for the chess agent.

Nothing in this module depends on a downloaded model.  A team-trained model is
enabled only when its file is included under ``weights/`` in the submission;
otherwise each component has a deterministic, cheap fallback.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import chess
import numpy as np

FEATURE_COUNT: Final = 2 * 6 * 64
DEFAULT_HIDDEN_SIZE: Final = 128
PIECE_VALUES: Final = (0, 100, 320, 330, 500, 900, 0)
PROMOTION_CODES: Final = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def _root_path(relative_path: str) -> Path | None:
    """Resolve a shipped relative path, refusing paths outside this module."""
    root = Path(__file__).resolve().parent
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def feature_index(piece: chess.Piece, square: chess.Square) -> int:
    """Return the sparse feature index for one piece on one square."""
    colour_offset = 6 if piece.color == chess.WHITE else 0
    return (colour_offset + piece.piece_type - 1) * 64 + square



def _fallback_evaluate(board: chess.Board) -> int:
    """A stable white-centric material fallback, expressed in centipawns."""
    score = 0
    for piece_type in range(chess.PAWN, chess.KING + 1):
        value = PIECE_VALUES[piece_type]
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))
    return score


@dataclass(slots=True)
class NNUEAccumulator:
    """Incrementally maintained first-layer activations for ``SparseNNUE``."""

    activations: np.ndarray

    def copy(self) -> NNUEAccumulator:
        """Return an independent accumulator suitable for a child search node."""
        return NNUEAccumulator(self.activations.copy())


class SparseNNUE:
    """A compact, team-trainable sparse evaluator.

    The optional ``.npz`` file contains quantised integer arrays:

    * ``feature_weights``: ``[768, hidden_size]`` int16;
    * ``hidden_bias``: ``[hidden_size]`` int32;
    * ``output_weights``: ``[hidden_size]`` int16; and
    * ``output_bias``: a scalar int32.

    Scores use ``output_scale`` (default 256).  This is NNUE-style rather than
    a copied Stockfish architecture: active piece-square features are summed in
    an accumulator, then evaluated with a small clipped-ReLU output layer.
    """

    def __init__(
        self,
        feature_weights: np.ndarray | None = None,
        hidden_bias: np.ndarray | None = None,
        output_weights: np.ndarray | None = None,
        output_bias: int = 0,
        output_scale: int = 256,
    ) -> None:
        self.feature_weights = feature_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias
        self.output_scale = max(1, output_scale)

    @property
    def enabled(self) -> bool:
        """Whether validated team-trained weights are available."""
        return self.feature_weights is not None

    @classmethod
    def from_shipped_weights(cls, relative_path: str = "weights/nnue.npz") -> SparseNNUE:
        """Load a bundled model, returning a deterministic disabled evaluator on failure."""
        path = _root_path(relative_path)
        if path is None or not path.is_file():
            return cls()
        try:
            with np.load(path, allow_pickle=False) as data:
                features = np.asarray(data["feature_weights"], dtype=np.int16)
                bias = np.asarray(data["hidden_bias"], dtype=np.int32)
                output = np.asarray(data["output_weights"], dtype=np.int16)
                output_bias = int(np.asarray(data["output_bias"]).item())
                scale = int(np.asarray(data.get("output_scale", 256)).item())
            if (
                features.ndim != 2
                or features.shape[0] != FEATURE_COUNT
                or bias.shape != (features.shape[1],)
                or output.shape != (features.shape[1],)
                or features.shape[1] == 0
            ):
                return cls()
            return cls(features, bias, output, output_bias, scale)
        except (KeyError, OSError, TypeError, ValueError):
            return cls()

    def accumulator(self, board: chess.Board) -> NNUEAccumulator:
        """Build an accumulator for ``board``; use ``apply_move`` for child nodes."""
        if not self.enabled:
            return NNUEAccumulator(np.empty(0, dtype=np.int32))
        assert self.hidden_bias is not None
        assert self.feature_weights is not None
        activations = self.hidden_bias.copy()
        for square, piece in board.piece_map().items():
            activations += self.feature_weights[feature_index(piece, square)]
        return NNUEAccumulator(activations)

    def apply_move(
        self, accumulator: NNUEAccumulator, board: chess.Board, move: chess.Move
    ) -> NNUEAccumulator:
        """Return a child accumulator after a legal move on the *pre-move* board.

        This does not mutate ``board`` or the parent accumulator.  Search code
        can call it before ``board.push(move)`` and carry the returned object.
        """
        if not self.enabled:
            return accumulator
        assert self.feature_weights is not None
        child = accumulator.copy()
        moving = board.piece_at(move.from_square)
        if moving is None:
            return child
        child.activations -= self.feature_weights[feature_index(moving, move.from_square)]

        captured_square = move.to_square
        if board.is_en_passant(move):
            captured_square += -8 if board.turn == chess.WHITE else 8
        captured = board.piece_at(captured_square)
        if captured is not None:
            child.activations -= self.feature_weights[feature_index(captured, captured_square)]

        placed = chess.Piece(move.promotion or moving.piece_type, moving.color)
        child.activations += self.feature_weights[feature_index(placed, move.to_square)]
        if board.is_castling(move):
            rook_from, rook_to = _castle_rook_squares(move)
            rook = board.piece_at(rook_from)
            if rook is not None:
                child.activations -= self.feature_weights[feature_index(rook, rook_from)]
                child.activations += self.feature_weights[feature_index(rook, rook_to)]
        return child

    def evaluate(self, board: chess.Board, accumulator: NNUEAccumulator | None = None) -> int:
        """Return a white-centric score in centipawns."""
        if not self.enabled:
            return _fallback_evaluate(board)
        if accumulator is None:
            accumulator = self.accumulator(board)
        assert self.output_weights is not None
        clipped = np.clip(accumulator.activations, 0, 127)
        total = int(np.dot(clipped.astype(np.int64), self.output_weights.astype(np.int64)))
        return (total + self.output_bias) // self.output_scale


def _castle_rook_squares(move: chess.Move) -> tuple[chess.Square, chess.Square]:
    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
        return move.to_square + 1, move.to_square - 1
    return move.to_square - 2, move.to_square + 1


def board_planes(board: chess.Board) -> np.ndarray:
    """Encode a position as 12 deterministic 8x8 piece planes for a policy model."""
    planes = np.zeros((1, 12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[0, plane, rank, file] = 1.0
    return planes


def policy_index(move: chess.Move) -> int:
    """Index in the 20,480-output policy convention used by this module."""
    promotion = PROMOTION_CODES[move.promotion]
    return ((promotion * 64 + move.from_square) * 64) + move.to_square


class ONNXPolicy:
    """Optional CPU-only policy ordering with a bounded position cache.

    A compatible team-trained ONNX model takes one ``float32`` input named
    ``board`` of shape ``[1, 12, 8, 8]`` and returns at least 20,480 flat logits
    under any output name.  No model means callers simply receive no policy
    scores and retain classical move ordering.
    """

    def __init__(self, relative_path: str = "weights/policy.onnx", cache_size: int = 512) -> None:
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, dict[chess.Move, float]] = OrderedDict()
        self._session: object | None = None
        self._input_name = "board"
        path = _root_path(relative_path)
        if path is None or not path.is_file():
            return
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            if input_info.name != "board":
                return
            self._session = session
            self._input_name = input_info.name
        except (ImportError, OSError, RuntimeError, ValueError):
            self._session = None

    @property
    def enabled(self) -> bool:
        return self._session is not None

    def score_moves(self, board: chess.Board, moves: list[chess.Move]) -> dict[chess.Move, float]:
        """Return logits for legal moves, or ``{}`` if inference is unavailable."""
        if self._session is None or not moves:
            return {}
        key = board.fen(en_passant="fen")
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return {move: cached[move] for move in moves if move in cached}
        try:
            session = self._session
            outputs = session.run(None, {self._input_name: board_planes(board)})  # type: ignore[attr-defined]
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if logits.size < 5 * 64 * 64:
                return {}
            scores = {move: float(logits[policy_index(move)]) for move in moves}
        except (IndexError, RuntimeError, TypeError, ValueError):
            return {}
        self._cache[key] = scores
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return scores
