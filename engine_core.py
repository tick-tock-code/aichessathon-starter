"""A readable, bounded, python-chess-correct alpha-beta searcher.

``SearchEngine.choose_move`` is intentionally the only integration point an agent needs.
It searches a supplied :class:`chess.Board` and returns a legal ``chess.Move``.  It retains
its transposition table and move-ordering heuristics across calls in one game.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Final, cast

import chess

from engine_eval import PIECE_VALUES, evaluate_for_side_to_move
from engine_time import SearchSelectivity, TimeBudget, TimeManager

INF: Final[int] = 1_000_000
MATE: Final[int] = 100_000
MAX_PLY: Final[int] = 96
type PositionKey = tuple[object, ...]
type Evaluator = Callable[[chess.Board], int]
type PolicyScorer = Callable[[chess.Board, list[chess.Move]], Mapping[chess.Move, float]]
type CancelCheck = Callable[[], bool]


class SearchTimeout(Exception):
    """Internal control flow used to abandon an incomplete iterative-deepening pass."""


@dataclass(slots=True)
class TTEntry:
    """A cached alpha-beta result; ``flag`` is exact, lower, or upper."""

    depth: int
    score: int
    flag: str
    best_move: chess.Move | None


@dataclass(slots=True)
class SearchStats:
    """Counters from the most recent call, useful for regression tests and profiling."""

    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    cutoffs: int = 0
    completed_depth: int = 0
    elapsed_ms: int = 0
    allocated_soft_ms: int = 0
    allocated_hard_ms: int = 0
    root_second_score: int = -INF
    root_best_score: int = -INF
    stable_iterations: int = 0
    best_move_changes: int = 0
    root_urgency: int = 0
    root_legal_moves: int = 0
    root_checking_moves: int = 0
    root_capturing_moves: int = 0
    root_scores_verified: bool = False
    iterations_started: int = 0
    iterations_completed: int = 0
    aborted_depth: int = 0
    last_iteration_ms: int = 0
    next_iteration_skipped: int = 0
    root_bound_gap: int = 0
    challenger_verifications: int = 0
    stop_reason: str = ""


@dataclass(slots=True)
class RootCandidate:
    """One root move's score and the strongest bound known for it."""

    move: chess.Move
    score: int
    upper_bound: int
    exact: bool


@dataclass(slots=True)
class RootSearchResult:
    """Completed root-search result, including challenger bounds for clock policy."""

    score: int
    move: chess.Move | None
    second_score: int
    candidates: list[RootCandidate]
    complete: bool

    @property
    def bound_gap(self) -> int:
        """Proved separation from the strongest alternative, or zero if unknown."""
        if not self.complete or self.move is None:
            return 0
        selected = next((item for item in self.candidates if item.move == self.move), None)
        if selected is None or not selected.exact:
            return 0
        alternatives = [item.upper_bound for item in self.candidates if item.move != self.move]
        if not alternatives:
            return INF
        return max(0, self.score - max(alternatives))

    def close_challengers(self, margin: int) -> list[RootCandidate]:
        """Return alternatives whose upper bound can still challenge the best move."""
        if self.move is None:
            return []
        return sorted(
            (
                item
                for item in self.candidates
                if item.move != self.move and item.upper_bound >= self.score - margin
            ),
            key=lambda item: (item.upper_bound, item.move.uci()),
            reverse=True,
        )


class SearchEngine:
    """Single-threaded iterative-deepening negamax with conservative selectivity."""

    def __init__(
        self,
        table_limit: int = 180_000,
        *,
        evaluator: Evaluator | None = None,
        policy: PolicyScorer | None = None,
        policy_max_ply: int = 2,
        time_manager: TimeManager | None = None,
        verify_root_scores: bool = False,
        time_manager_mode: str = "legacy",
    ) -> None:
        self.table_limit = table_limit
        self.evaluator = evaluator or evaluate_for_side_to_move
        self.policy = policy
        self.policy_max_ply = max(0, policy_max_ply)
        self.time_manager = time_manager or TimeManager()
        self.verify_root_scores = verify_root_scores
        self.time_manager_mode = (
            time_manager_mode if time_manager_mode in {"legacy", "guarded"} else "legacy"
        )
        self.tt: dict[PositionKey, TTEntry] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.stats = SearchStats()
        self._deadline = 0.0
        self._hard_deadline = 0.0
        self._cancel: CancelCheck | None = None
        self._selectivity = SearchSelectivity(3, 4, 2, 3, 2, 110, 120, 8)

    def choose_move(
        self,
        board: chess.Board,
        time_left_ms: int,
        *,
        max_depth: int = 64,
        move_time_ms: int | None = None,
        cancel: CancelCheck | None = None,
    ) -> chess.Move:
        """Return a legal move, retaining the last fully searched iteration on timeout.

        ``move_time_ms`` is an optional test/tournament override.  The default is deliberately
        conservative: it preserves clock headroom for the Python runtime and later game phases.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("choose_move requires a position with at least one legal move")
        fallback = legal_moves[0]
        self.stats = SearchStats()
        self._cancel = cancel
        start = perf_counter()
        if move_time_ms is None:
            budget = self.time_manager.initial_budget(board, time_left_ms)
        else:
            allocation = max(1, min(move_time_ms, max(1, time_left_ms - 10)))
            signals = self.time_manager.root_signals(board)
            budget = TimeBudget(
                allocation,
                allocation,
                signals,
                self.time_manager.selectivity_for(time_left_ms, signals.urgency),
            )
        self._selectivity = budget.selectivity
        self._deadline = start + budget.soft_ms / 1_000.0
        self._hard_deadline = start + budget.hard_ms / 1_000.0
        self.stats.allocated_soft_ms = budget.soft_ms
        self.stats.allocated_hard_ms = budget.hard_ms
        self.stats.root_urgency = budget.signals.urgency
        self.stats.root_legal_moves = budget.signals.legal_moves
        self.stats.root_checking_moves = budget.signals.checking_moves
        self.stats.root_capturing_moves = budget.signals.capturing_moves

        best_move = fallback
        best_score = -INF
        previous_move: chess.Move | None = None
        stable_iterations = 0
        last_iteration_ms = 0
        aspiration = 45
        for depth in range(1, max_depth + 1):
            elapsed_before_iteration = int((perf_counter() - start) * 1_000)
            if (
                self.time_manager_mode == "guarded"
                and depth >= 3
                and not self._can_finish_next_iteration(
                    elapsed_before_iteration, budget.hard_ms, last_iteration_ms
                )
            ):
                self.stats.next_iteration_skipped += 1
                self.stats.stop_reason = "next_iteration_would_timeout"
                break
            iteration_start = perf_counter()
            self.stats.iterations_started += 1
            try:
                if depth == 1 or self.verify_root_scores:
                    result = self._root_search(board, depth, -INF, INF)
                else:
                    alpha = max(-INF, best_score - aspiration)
                    beta = min(INF, best_score + aspiration)
                    result = self._root_search(board, depth, alpha, beta)
                    if result.score <= alpha or result.score >= beta:
                        result = self._root_search(board, depth, -INF, INF)
                score, move, second_score = result.score, result.move, result.second_score
                if move is not None:
                    best_move, best_score = move, score
                    self.stats.completed_depth = depth
                    self.stats.iterations_completed += 1
                    last_iteration_ms = int((perf_counter() - iteration_start) * 1_000)
                    self.stats.last_iteration_ms = last_iteration_ms
                    self.stats.root_best_score = best_score
                    self.stats.root_second_score = second_score
                    self.stats.root_scores_verified = self.verify_root_scores
                    self.stats.root_bound_gap = result.bound_gap
                    stable_iterations = stable_iterations + 1 if move == previous_move else 0
                    if previous_move is not None and move != previous_move:
                        self.stats.best_move_changes += 1
                    previous_move = move
                    self.stats.stable_iterations = stable_iterations
                aspiration = min(300, aspiration + 8)
                if abs(best_score) >= MATE - MAX_PLY:
                    self.stats.stop_reason = "mate"
                    break
                elapsed_ms = int((perf_counter() - start) * 1_000)
                if self._at_or_past_soft_deadline():
                    if self.time_manager_mode == "guarded":
                        if stable_iterations >= 2 and result.bound_gap >= 35:
                            self.stats.stop_reason = "proven_root_gap"
                            break
                        if stable_iterations >= 2 and self._verify_root_challengers(
                            board, depth, result
                        ):
                            self.stats.root_bound_gap = result.bound_gap
                            self.stats.stop_reason = "verified_root_gap"
                            break
                    elif TimeManager.should_stop_after_iteration(
                        elapsed_ms,
                        budget,
                        stable_iterations,
                        max(0, best_score - second_score) if self.verify_root_scores else 0,
                        False,
                    ):
                        self.stats.stop_reason = "legacy_root_gap"
                        break
            except SearchTimeout:
                self.stats.aborted_depth = depth
                self.stats.stop_reason = "hard_deadline"
                break
        self.stats.elapsed_ms = int((perf_counter() - start) * 1_000)
        return best_move

    @staticmethod
    def _can_finish_next_iteration(
        elapsed_ms: int, hard_ms: int, previous_iteration_ms: int
    ) -> bool:
        """Conservatively decide whether another completed depth can fit."""
        projected_ms = max(20, int(previous_iteration_ms * 2.25))
        return elapsed_ms + projected_ms + 10 < hard_ms

    def _verify_root_challengers(
        self, board: chess.Board, depth: int, result: RootSearchResult
    ) -> bool:
        """Spend a bounded post-soft slice proving only close root alternatives."""
        if result.move is None or result.bound_gap >= 35:
            return result.bound_gap >= 35
        candidates = result.close_challengers(35)[:2]
        if not candidates:
            return False
        remaining_ms = int((self._hard_deadline - perf_counter()) * 1_000)
        cap_ms = min(200, max(25, remaining_ms // 3))
        if cap_ms < 25:
            return False
        original_deadline = self._hard_deadline
        self._hard_deadline = min(original_deadline, perf_counter() + cap_ms / 1_000.0)
        try:
            for candidate in candidates:
                board.push(candidate.move)
                try:
                    candidate.score = -self._search(board, depth - 1, -INF, INF, 1, True)
                    candidate.upper_bound = candidate.score
                    candidate.exact = True
                    self.stats.challenger_verifications += 1
                finally:
                    board.pop()
        except SearchTimeout:
            return False
        finally:
            self._hard_deadline = original_deadline
        return result.bound_gap >= 35

    def _root_search(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> RootSearchResult:
        self._check_deadline()
        original_alpha = alpha
        key = self._key(board)
        entry = self.tt.get(key)
        tt_move = entry.best_move if entry is not None else None
        moves = self._ordered_moves(board, tt_move, 0)
        best_move: chess.Move | None = None
        best_score = -INF
        second_score = -INF
        candidates: list[RootCandidate] = []
        complete = True
        for index, move in enumerate(moves):
            alpha_before = alpha
            exact = index == 0 or self.verify_root_scores
            board.push(move)
            try:
                if exact:
                    score = -self._search(board, depth - 1, -beta, -alpha, 1, True)
                else:
                    # Principal variation search: cheap zero-window test before a full re-search.
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, 1, True)
                        exact = True
            finally:
                board.pop()
            if not (alpha_before < score < beta or (alpha_before == -INF and beta == INF)):
                exact = False
            upper_bound = score if exact or score <= alpha_before else INF
            candidates.append(RootCandidate(move, score, upper_bound, exact))
            if score > best_score:
                second_score = best_score
                best_score, best_move = score, move
            elif score > second_score:
                second_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.stats.cutoffs += 1
                complete = False
                break
        flag = "exact" if best_score > original_alpha and best_score < beta else "upper"
        if best_score >= beta:
            flag = "lower"
        self._store(key, TTEntry(depth, best_score, flag, best_move))
        return RootSearchResult(best_score, best_move, second_score, candidates, complete)

    def _search(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool
    ) -> int:
        self.stats.nodes += 1
        if self.stats.nodes & 7 == 0:
            self._check_deadline()
        if ply >= MAX_PLY:
            return self._evaluate(board)
        if board.is_repetition(3) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate():
            return 0

        key = self._key(board)
        entry = self.tt.get(key)
        tt_move: chess.Move | None = None
        if entry is not None:
            self.stats.tt_hits += 1
            tt_move = entry.best_move
            if entry.depth >= depth:
                if entry.flag == "exact":
                    return entry.score
                if entry.flag == "lower":
                    alpha = max(alpha, entry.score)
                elif entry.flag == "upper":
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        in_check = board.is_check()
        static_score = self._evaluate(board)
        # Null move pruning is gated by non-pawn material to avoid common pawn-endgame zugzwangs.
        if (
            allow_null
            and depth >= self._selectivity.null_min_depth
            and not in_check
            and static_score >= beta
            and self._has_non_pawn_material(board, board.turn)
        ):
            board.push(chess.Move.null())
            try:
                reduction = self._selectivity.null_reduction(depth)
                null_score = -self._search(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
                )
            finally:
                board.pop()
            if null_score >= beta:
                self.stats.cutoffs += 1
                return null_score

        original_alpha = alpha
        moves = self._ordered_moves(board, tt_move, ply)
        if not moves:
            return -MATE + ply if in_check else 0
        best_score = -INF
        best_move: chess.Move | None = None
        for index, move in enumerate(moves):
            quiet = not board.is_capture(move) and move.promotion is None
            # A very modest futility margin at the final pre-quiescence ply.
            if (
                depth == 1
                and quiet
                and not in_check
                and static_score + self._selectivity.futility_margin <= alpha
            ):
                continue
            board.push(move)
            try:
                reduction = self._selectivity.lmr_reduction(depth, index, quiet, in_check)
                if index == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    score = -self._search(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha and reduction:
                        score = -self._search(board, depth - 1, -alpha - 1, -alpha, ply + 1, True)
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, ply + 1, True)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.stats.cutoffs += 1
                if quiet:
                    self._record_quiet_cutoff(move, ply, depth, board.turn)
                break

        if best_move is None:  # All moves were safely futile.
            return static_score
        flag = "exact" if best_score > original_alpha and best_score < beta else "upper"
        if best_score >= beta:
            flag = "lower"
        self._store(key, TTEntry(depth, best_score, flag, best_move))
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, qdepth: int) -> int:
        self.stats.qnodes += 1
        if self.stats.qnodes & 7 == 0:
            self._check_deadline()
        if board.is_repetition(3) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate():
            return 0
        in_check = board.is_check()
        if qdepth >= self._selectivity.qdepth_limit:
            return self._evaluate(board)
        stand_pat = self._evaluate(board)
        if not in_check:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)

        moves = (
            list(board.legal_moves)
            if in_check
            else [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]
        )
        moves.sort(key=lambda move: self._capture_score(board, move), reverse=True)
        for move in moves:
            # Delta pruning only applies to ordinary captures, never checks/evasions/promotions.
            if not in_check and move.promotion is None:
                victim = board.piece_at(move.to_square)
                victim_value = PIECE_VALUES[victim.piece_type] if victim is not None else 100
                if stand_pat + victim_value + self._selectivity.delta_margin < alpha:
                    continue
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        policy_rank: dict[chess.Move, int] = {}
        if self.policy is not None and ply <= self.policy_max_ply and len(moves) > 1:
            try:
                scores = self.policy(board, moves)
                ranked = sorted(moves, key=lambda move: scores.get(move, float("-inf")))
                policy_rank = {move: rank * 64 for rank, move in enumerate(ranked, start=1)}
            except (RuntimeError, TypeError, ValueError, OverflowError):
                policy_rank = {}
        moves.sort(
            key=lambda move: self._move_score(board, move, tt_move, ply)
            + policy_rank.get(move, 0),
            reverse=True,
        )
        return moves

    def _evaluate(self, board: chess.Board) -> int:
        """Evaluate from the side-to-move perspective through the selected backend."""
        return int(self.evaluator(board))

    def _move_score(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return 10_000_000
        if board.is_capture(move) or move.promotion is not None:
            return 1_000_000 + self._capture_score(board, move)
        killer_one, killer_two = self.killers[min(ply, MAX_PLY - 1)]
        if move == killer_one:
            return 900_000
        if move == killer_two:
            return 800_000
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    @staticmethod
    def _capture_score(board: chess.Board, move: chess.Move) -> int:
        """MVV-LVA ordering: a cheap and safe stand-in for full static exchange evaluation."""
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = PIECE_VALUES[victim.piece_type] if victim is not None else 100
        attacker_value = PIECE_VALUES[attacker.piece_type] if attacker is not None else 0
        promotion = PIECE_VALUES[move.promotion] if move.promotion is not None else 0
        return victim_value * 16 - attacker_value + promotion

    @staticmethod
    def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
        pieces = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        return any(board.pieces(piece, color) for piece in pieces)

    @staticmethod
    def _key(board: chess.Board) -> PositionKey:
        """Compact python-chess identity plus the 50-move draw state.

        The library's internal key contains every rule-relevant position field
        except the halfmove counter. Keeping the complete tuple avoids hash
        collisions while being far cheaper than rebuilding a FEN at every node.
        """
        key = cast(tuple[object, ...], board._transposition_key())
        return (*key, min(board.halfmove_clock, 100))

    def _store(self, key: PositionKey, entry: TTEntry) -> None:
        if len(self.tt) >= self.table_limit:
            # Clearing is deterministic and bounds memory; a later upgrade can use aging.
            self.tt.clear()
        old = self.tt.get(key)
        if old is None or entry.depth >= old.depth:
            self.tt[key] = entry

    def _record_quiet_cutoff(
        self, move: chess.Move, ply: int, depth: int, color: chess.Color
    ) -> None:
        slot = self.killers[min(ply, MAX_PLY - 1)]
        if move != slot[0]:
            slot[1] = slot[0]
            slot[0] = move
        key = (color, move.from_square, move.to_square)
        self.history[key] = min(32_000, self.history.get(key, 0) + depth * depth)

    def _check_deadline(self) -> None:
        if (self._cancel is not None and self._cancel()) or perf_counter() >= self._hard_deadline:
            raise SearchTimeout

    def _at_or_past_soft_deadline(self) -> bool:
        return perf_counter() >= self._deadline
