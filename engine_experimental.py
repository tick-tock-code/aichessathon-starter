"""Small, optional experiments for non-alpha-beta chess search.

This module is deliberately independent of :mod:`agent`.  It provides three
experiments that can be compared behind the same ``chess.Board`` interface:

* :class:`PuctSearch` is a single-core MCTS/PUCT search.  Its node table is a
  conservative transposition DAG; the exact FEN (including draw counters) is
  used as the key and each simulation also guards against cycles.
* :class:`SquareTokenPolicy` is an inference hook for a small square-token
  policy/transformer.  A model is optional: the fallback is deterministic and
  scores forcing, central and developing moves.
* :class:`PhaseExpertRouter` and :class:`ImplicitSearch` provide a compact
  mixture-of-experts and an action-value (implicit-search) baseline.

The callbacks are intentionally tiny and use only Python and python-chess.
Any neural weights or model runtime can be added by the caller, provided the
competition's packaging rules are respected.  No callback is invoked after
the search deadline can be observed by this module, although a callback that
itself blocks cannot be interrupted by Python.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import chess

WHITE = chess.WHITE
BLACK = chess.BLACK
MATE_VALUE = 1.0

PolicyFn = Callable[[chess.Board, Sequence[chess.Move]], Mapping[chess.Move, float]]
ValueFn = Callable[[chess.Board], float]
ActionValueFn = Callable[[chess.Board, chess.Move], float]
TokenInference = Callable[[tuple[int, ...], tuple[str, ...]], Sequence[float]]
ExpertFn = Callable[[chess.Board], float]


def position_key(board: chess.Board) -> str:
    """Return a conservative key suitable for transposition sharing.

    The full FEN retains side to move, castling, en-passant and both draw
    counters.  It is a little less compact than a Zobrist key, but avoids
    collisions and keeps this experimental implementation easy to audit.
    Repetition history is still path-dependent, so PUCT separately detects a
    repeated key on each simulation path.
    """

    return board.fen(en_passant="fen")


def _piece_value(piece_type: int) -> float:
    return {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }.get(piece_type, 0.0)


def _material_white(board: chess.Board) -> float:
    score = 0.0
    for piece_type in range(chess.PAWN, chess.KING + 1):
        value = _piece_value(piece_type)
        score += value * len(board.pieces(piece_type, WHITE))
        score -= value * len(board.pieces(piece_type, BLACK))
    return score


def _normalise_white_score(score: float) -> float:
    """Map a centipawn-like score to a useful MCTS value in ``[-1, 1]``."""

    if not math.isfinite(score):
        return 0.0
    return max(-1.0, min(1.0, math.tanh(score / 8.0)))


def _fallback_policy(board: chess.Board, moves: Sequence[chess.Move]) -> Mapping[chess.Move, float]:
    """Cheap deterministic policy used when no learned policy is supplied."""

    scores: dict[chess.Move, float] = {}
    mover = board.turn
    for move in moves:
        score = 0.0
        captured = board.piece_at(move.to_square)
        moving = board.piece_at(move.from_square)
        if captured is not None:
            score += 1.5 * _piece_value(captured.piece_type)
        if board.is_en_passant(move):
            score += 1.5
        if move.promotion is not None:
            score += 8.0 + _piece_value(move.promotion)
        if moving is not None and moving.piece_type == chess.KING and board.is_castling(move):
            score += 0.8
        if board.gives_check(move):
            score += 1.25
        file_index = chess.square_file(move.to_square)
        rank_index = chess.square_rank(move.to_square)
        centre_distance = abs(3.5 - file_index) + abs(3.5 - rank_index)
        score += 0.12 * (7.0 - centre_distance)
        if moving is not None and moving.color == mover and moving.piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
        ):
            score += 0.15
        scores[move] = score
    return scores


def _policy_priors(
    board: chess.Board,
    moves: Sequence[chess.Move],
    policy: PolicyFn | None,
) -> dict[chess.Move, float]:
    raw = _fallback_policy(board, moves) if policy is None else policy(board, moves)
    logits = [float(raw.get(move, 0.0)) for move in moves]
    finite = [value if math.isfinite(value) else 0.0 for value in logits]
    if not finite:
        return {}
    maximum = max(finite)
    exponentials = [math.exp(max(-50.0, min(50.0, value - maximum))) for value in finite]
    total = sum(exponentials)
    if total <= 0.0 or not math.isfinite(total):
        prior = 1.0 / len(moves)
        return {move: prior for move in moves}
    return {
        move: exp_value / total
        for move, exp_value in zip(moves, exponentials, strict=True)
    }


def _terminal_value(board: chess.Board) -> float | None:
    """Return the terminal value from White's perspective."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return MATE_VALUE if outcome.winner == WHITE else -MATE_VALUE


@dataclass(slots=True)
class _Edge:
    move: chess.Move
    prior: float
    child_key: str
    visits: int = 0


@dataclass(slots=True)
class _DAGNode:
    key: str
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[str, _Edge] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class TranspositionDAG:
    """Reusable state table for PUCT experiments.

    Nodes are shared by exact-FEN key.  Edge objects remain parent-specific,
    which is important: the same child position may be reached by different
    moves and those edges need separate visit counts.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, _DAGNode] = {}

    def node(self, key: str) -> _DAGNode:
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        created = _DAGNode(key)
        self.nodes[key] = created
        return created

    def clear(self) -> None:
        self.nodes.clear()


class PuctSearch:
    """Deadline-aware, single-core PUCT with optional policy/value callbacks.

    ``value`` must return a number in ``[-1, 1]`` from White's perspective;
    values outside that range are clipped.  ``policy`` returns move logits or
    arbitrary real scores, one mapping entry per legal move.  Omitting either
    callback selects the deterministic classical fallback.
    """

    def __init__(
        self,
        policy: PolicyFn | None = None,
        value: ValueFn | None = None,
        exploration: float = 1.35,
    ) -> None:
        self.policy = policy
        self.value = value or self._default_value
        self.exploration = max(0.05, exploration)
        self.dag = TranspositionDAG()

    def _default_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -MATE_VALUE if board.turn == WHITE else MATE_VALUE
        return _normalise_white_score(_material_white(board))

    def _expand(self, node: _DAGNode, board: chess.Board) -> None:
        moves = list(board.legal_moves)
        if not moves:
            node.expanded = True
            return
        priors = _policy_priors(board, moves, self.policy)
        for move in moves:
            board.push(move)
            child_key = position_key(board)
            board.pop()
            node.children[move.uci()] = _Edge(move, priors.get(move, 0.0), child_key)
            self.dag.node(child_key)
        node.expanded = True

    def _select(self, node: _DAGNode, board: chess.Board) -> _Edge:
        # Give every legal edge one visit before trusting a heuristic/value
        # estimate. Without this first-play urgency, one superficially good
        # policy move can monopolise a short CPU search and hide mate in one.
        unvisited = [edge for edge in node.children.values() if edge.visits == 0]
        if unvisited:
            return max(unvisited, key=lambda edge: (edge.prior, edge.move.uci()))
        parent_visits = max(1, node.visits)
        maximizing = board.turn == WHITE
        best_edge: _Edge | None = None
        best_score = -math.inf
        for edge in node.children.values():
            child = self.dag.node(edge.child_key)
            exploitation = child.mean_value if maximizing else -child.mean_value
            exploration = self.exploration * edge.prior * math.sqrt(parent_visits) / (
                1.0 + edge.visits
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_edge = edge
        if best_edge is None:
            raise RuntimeError("PUCT selection reached an empty non-terminal node")
        return best_edge

    def _evaluate_leaf(self, board: chess.Board) -> float:
        terminal = _terminal_value(board)
        if terminal is not None:
            return terminal
        estimate = float(self.value(board))
        if not math.isfinite(estimate):
            return 0.0
        # Callbacks and stored DAG values are White-relative. Selection flips
        # exploitation at Black-to-move nodes, including a Black root.
        return max(-MATE_VALUE, min(MATE_VALUE, estimate))

    def choose(self, board: chess.Board, time_limit_ms: int) -> chess.Move:
        """Run simulations until the budget expires and return a legal move."""

        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("cannot choose a move from a terminal position")
        root_key = position_key(board)
        self.dag.clear()
        root = self.dag.node(root_key)
        self._expand(root, board)
        deadline = time.monotonic() + max(0.0, float(time_limit_ms)) / 1000.0
        simulations = 0
        while time.monotonic() < deadline or simulations == 0:
            state = board.copy(stack=True)
            current = root
            visited_keys = {root_key}
            path_nodes = [root]
            path_edges: list[_Edge] = []
            value: float | None = None
            while True:
                terminal = _terminal_value(state)
                if terminal is not None:
                    value = terminal
                    break
                if not current.expanded:
                    self._expand(current, state)
                    value = self._evaluate_leaf(state)
                    break
                if not current.children:
                    value = self._evaluate_leaf(state)
                    break
                edge = self._select(current, state)
                state.push(edge.move)
                path_edges.append(edge)
                child = self.dag.node(edge.child_key)
                path_nodes.append(child)
                if edge.child_key in visited_keys:
                    value = self._evaluate_leaf(state)
                    break
                visited_keys.add(edge.child_key)
                current = child
            assert value is not None
            for node in path_nodes:
                node.visits += 1
                node.value_sum += value
            for edge in path_edges:
                edge.visits += 1
            simulations += 1
        # Visits are the robust MCTS choice; policy score is the deterministic
        # tie-breaker and ensures a useful result if only one simulation ran.
        selected = max(
            root.children.values(),
            key=lambda edge: (edge.visits, edge.prior),
        )
        return selected.move


def encode_square_tokens(board: chess.Board) -> tuple[int, ...]:
    """Encode a board as 64 square tokens for a small transformer hook.

    Token IDs are 0 for empty, 1--6 for White pawn--king and 7--12 for Black
    pawn--king.  Metadata tokens follow the board: side to move, four castling
    flags, and an en-passant square (19 means none).  Keeping this function
    dependency-free makes it suitable for a caller's Torch or ONNX adapter.
    """

    tokens = [0] * 64
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color == WHITE else 6
        tokens[square] = offset + piece.piece_type
    tokens.append(13 if board.turn == WHITE else 14)
    tokens.extend(
        [
            15 if board.has_kingside_castling_rights(WHITE) else 0,
            16 if board.has_queenside_castling_rights(WHITE) else 0,
            17 if board.has_kingside_castling_rights(BLACK) else 0,
            18 if board.has_queenside_castling_rights(BLACK) else 0,
        ]
    )
    tokens.append(20 + board.ep_square if board.ep_square is not None else 19)
    return tuple(tokens)


class SquareTokenPolicy:
    """Search-free policy adapter with a deterministic fallback.

    A trained model can be supplied as ``inference(tokens, legal_uci)`` and
    should return one logit per legal UCI move.  Restricting the hook to legal
    moves lets tiny models avoid a 64x64 output head and makes every output
    safe to hand to the competition runner.
    """

    def __init__(self, inference: TokenInference | None = None) -> None:
        self.inference = inference

    def scores(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
    ) -> Mapping[chess.Move, float]:
        if not moves:
            return {}
        fallback = _fallback_policy(board, moves)
        if self.inference is None:
            return fallback
        tokens = encode_square_tokens(board)
        legal_uci = tuple(move.uci() for move in moves)
        try:
            output = tuple(float(value) for value in self.inference(tokens, legal_uci))
        except (TypeError, ValueError, OverflowError):
            return fallback
        if len(output) != len(moves):
            return fallback
        return {
            move: value if math.isfinite(value) else fallback[move]
            for move, value in zip(moves, output, strict=True)
        }

    def choose(self, board: chess.Board, time_limit_ms: int = 1) -> chess.Move:
        """Choose the highest policy-scored legal move within the tiny budget."""

        del time_limit_ms  # The adapter performs one bounded inference call.
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("cannot choose a move from a terminal position")
        scores = self.scores(board, moves)
        return max(moves, key=lambda move: (scores.get(move, -math.inf), move.uci()))


def _opening_expert(board: chess.Board) -> float:
    mobility = len(list(board.legal_moves))
    return _material_white(board) + 0.08 * (mobility if board.turn == WHITE else -mobility)


def _middlegame_expert(board: chess.Board) -> float:
    mobility = len(list(board.legal_moves))
    black_king = board.king(BLACK)
    white_king = board.king(WHITE)
    king_safety = 0.0
    if black_king is not None and white_king is not None:
        king_safety = 0.25 * (
            len(board.attackers(WHITE, black_king)) - len(board.attackers(BLACK, white_king))
        )
    turn_mobility = mobility if board.turn == WHITE else -mobility
    return _material_white(board) + 0.04 * turn_mobility + king_safety


def _endgame_expert(board: chess.Board) -> float:
    score = _material_white(board)
    for color, sign in ((WHITE, 1.0), (BLACK, -1.0)):
        for square in board.pieces(chess.PAWN, color):
            rank = chess.square_rank(square)
            score += sign * (0.08 * (rank if color == WHITE else 7 - rank))
    return score


class PhaseExpertRouter:
    """Route a position to opening, middlegame or endgame evaluators."""

    def __init__(
        self,
        opening: ExpertFn | None = None,
        middlegame: ExpertFn | None = None,
        endgame: ExpertFn | None = None,
    ) -> None:
        self.experts: dict[str, ExpertFn] = {
            "opening": opening or _opening_expert,
            "middlegame": middlegame or _middlegame_expert,
            "endgame": endgame or _endgame_expert,
        }

    def phase(self, board: chess.Board) -> str:
        non_pawns = len(board.pieces(chess.KNIGHT, WHITE)) + len(board.pieces(chess.KNIGHT, BLACK))
        non_pawns += len(board.pieces(chess.BISHOP, WHITE)) + len(board.pieces(chess.BISHOP, BLACK))
        non_pawns += len(board.pieces(chess.ROOK, WHITE)) + len(board.pieces(chess.ROOK, BLACK))
        non_pawns += len(board.pieces(chess.QUEEN, WHITE)) + len(board.pieces(chess.QUEEN, BLACK))
        if board.fullmove_number <= 10 and non_pawns >= 8:
            return "opening"
        pawns = len(board.pieces(chess.PAWN, WHITE)) + len(board.pieces(chess.PAWN, BLACK))
        if non_pawns <= 4 or pawns <= 6:
            return "endgame"
        return "middlegame"

    def evaluate(self, board: chess.Board, perspective: chess.Color = WHITE) -> float:
        score = float(self.experts[self.phase(board)](board))
        score = _normalise_white_score(score)
        return score if perspective == WHITE else -score


class ImplicitSearch:
    """Select moves using a learned/action-value callback without a tree.

    ``action_value(board, move)`` should estimate the resulting position from
    White's perspective.  This is useful for a compact value/policy model that
    internally predicts a few future steps (implicit search), while retaining
    a deterministic router fallback for experiments without weights.
    """

    def __init__(
        self,
        action_value: ActionValueFn | None = None,
        policy: SquareTokenPolicy | None = None,
        router: PhaseExpertRouter | None = None,
    ) -> None:
        self.policy = policy or SquareTokenPolicy()
        self.router = router or PhaseExpertRouter()
        self.action_value = action_value or self._default_action_value

    def _default_action_value(self, board: chess.Board, move: chess.Move) -> float:
        board.push(move)
        value = self.router.evaluate(board)
        board.pop()
        return value

    def choose(self, board: chess.Board, time_limit_ms: int) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("cannot choose a move from a terminal position")
        deadline = time.monotonic() + max(0.0, float(time_limit_ms)) / 1000.0
        scores = self.policy.scores(board, legal)
        best_move = max(legal, key=lambda move: (scores.get(move, -math.inf), move.uci()))
        best_score = -math.inf if board.turn == WHITE else math.inf
        attempted = False
        for move in legal:
            if attempted and time.monotonic() >= deadline:
                break
            attempted = True
            try:
                score = float(self.action_value(board, move))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(score):
                continue
            if (board.turn == WHITE and score > best_score) or (
                board.turn == BLACK and score < best_score
            ):
                best_move = move
                best_score = score
        return best_move


def mcts_move(
    fen: str,
    time_left_ms: int,
    policy: PolicyFn | None = None,
    value: ValueFn | None = None,
) -> str:
    """Convenience entry point returning a legal UCI move for a FEN."""

    board = chess.Board(fen)
    return PuctSearch(policy=policy, value=value).choose(board, time_left_ms).uci()


__all__ = [
    "ActionValueFn",
    "ExpertFn",
    "ImplicitSearch",
    "PhaseExpertRouter",
    "PolicyFn",
    "PuctSearch",
    "SquareTokenPolicy",
    "TokenInference",
    "TranspositionDAG",
    "ValueFn",
    "encode_square_tokens",
    "mcts_move",
    "position_key",
]
