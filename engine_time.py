"""Clock allocation and conservative search-selectivity policy.

This module contains no chess search itself.  Keeping the policy separate makes
it straightforward to tune with a tournament harness and keeps the actual
searcher readable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chess


@dataclass(frozen=True, slots=True)
class RootSignals:
    """Cheap tactical facts measured before the iterative-deepening search."""

    in_check: bool
    legal_moves: int
    checking_moves: int
    capturing_moves: int

    @property
    def urgency(self) -> int:
        """Return a small bounded urgency score; larger means more forcing."""
        score = 0
        if self.in_check:
            score += 4
        if self.legal_moves <= 3:
            score += 3
        elif self.legal_moves <= 8:
            score += 1
        if self.checking_moves >= 2:
            score += 2
        elif self.checking_moves:
            score += 1
        if self.capturing_moves >= 5:
            score += 1
        return min(score, 8)


@dataclass(frozen=True, slots=True)
class SearchSelectivity:
    """The pruning settings for one search; safer settings prune less."""

    lmr_min_depth: int
    lmr_move_number: int
    lmr_max_reduction: int
    null_min_depth: int
    null_base_reduction: int
    futility_margin: int
    delta_margin: int
    qdepth_limit: int

    def lmr_reduction(self, depth: int, move_index: int, quiet: bool, in_check: bool) -> int:
        if (
            not quiet
            or in_check
            or depth < self.lmr_min_depth
            or move_index < self.lmr_move_number
        ):
            return 0
        return min(self.lmr_max_reduction, 1 + int(move_index >= 10 and depth >= 5))

    def null_reduction(self, depth: int) -> int:
        return self.null_base_reduction + depth // 5


@dataclass(frozen=True, slots=True)
class TimeBudget:
    """Soft and hard per-move limits, measured in milliseconds."""

    soft_ms: int
    hard_ms: int
    signals: RootSignals
    selectivity: SearchSelectivity


class TimeManager:
    """A bounded time allocator for the fixed-increment Chessathon clock."""

    increment_ms = 500
    minimum_ms = 25
    maximum_soft_ms = 5_000

    @staticmethod
    def root_signals(board: chess.Board) -> RootSignals:
        moves = list(board.legal_moves)
        checks = 0
        captures = 0
        for move in moves:
            captures += int(board.is_capture(move))
            board.push(move)
            try:
                checks += int(board.is_check())
            finally:
                board.pop()
        return RootSignals(board.is_check(), len(moves), checks, captures)

    @staticmethod
    def _estimated_moves_remaining(board: chess.Board) -> int:
        # Material is a robust phase proxy, unlike a ply counter which is not
        # available in an incoming FEN.  Bound the estimate to avoid both
        # reckless opening spends and absurdly small endgame allocations.
        non_king_pieces = sum(
            len(board.pieces(piece, color))
            for color in (chess.WHITE, chess.BLACK)
            for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        )
        return max(12, min(34, 10 + non_king_pieces))

    @staticmethod
    def selectivity_for(time_left_ms: int, urgency: int) -> SearchSelectivity:
        # Tactical positions deliberately choose the safer profile: additional
        # clock is converted into verification, not merely more speculation.
        if time_left_ms < 3_000:
            return SearchSelectivity(3, 4, 2, 3, 2, 85, 95, 7)
        if urgency >= 4:
            return SearchSelectivity(4, 8, 1, 4, 2, 175, 180, 10)
        return SearchSelectivity(3, 4, 2, 3, 2, 110, 120, 8)

    def __init__(self, profile: str = "balanced") -> None:
        self.profile = (
            profile if profile in {"very_fast", "fast", "balanced", "safe"} else "balanced"
        )
        try:
            requested_scale = float(os.environ.get("CHESSATHON_TIME_SCALE", "1.0"))
        except ValueError:
            requested_scale = 1.0
        self.time_scale = max(0.25, min(1.5, requested_scale))
        requested_selectivity = os.environ.get("CHESSATHON_SELECTIVITY", "profile")
        self.selectivity_override = (
            requested_selectivity
            if requested_selectivity in {"profile", "aggressive", "safe"}
            else "profile"
        )

    @staticmethod
    def _env_selectivity_int(name: str, default: int, minimum: int, maximum: int) -> int:
        """Read a bounded developer-only pruning override without changing defaults."""
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

    def _apply_selectivity_overrides(self, selectivity: SearchSelectivity) -> SearchSelectivity:
        """Apply explicit local-tuning values after choosing a coherent base preset."""
        return SearchSelectivity(
            self._env_selectivity_int(
                "CHESSATHON_LMR_MIN_DEPTH", selectivity.lmr_min_depth, 1, 12
            ),
            self._env_selectivity_int(
                "CHESSATHON_LMR_MOVE_NUMBER", selectivity.lmr_move_number, 1, 32
            ),
            self._env_selectivity_int(
                "CHESSATHON_LMR_MAX_REDUCTION", selectivity.lmr_max_reduction, 0, 4
            ),
            self._env_selectivity_int(
                "CHESSATHON_NULL_MIN_DEPTH", selectivity.null_min_depth, 1, 12
            ),
            self._env_selectivity_int(
                "CHESSATHON_NULL_BASE_REDUCTION", selectivity.null_base_reduction, 1, 4
            ),
            self._env_selectivity_int(
                "CHESSATHON_FUTILITY_MARGIN", selectivity.futility_margin, 0, 300
            ),
            self._env_selectivity_int(
                "CHESSATHON_DELTA_MARGIN", selectivity.delta_margin, 0, 300
            ),
            self._env_selectivity_int(
                "CHESSATHON_QDEPTH_LIMIT", selectivity.qdepth_limit, 1, 16
            ),
        )

    def initial_budget(self, board: chess.Board, time_left_ms: int) -> TimeBudget:
        signals = self.root_signals(board)
        remaining = max(1, time_left_ms)
        reserve = min(750, max(30, remaining // 16))
        spendable = max(1, remaining - reserve)
        expected_moves = self._estimated_moves_remaining(board)
        base = remaining // expected_moves + self.increment_ms // 3
        phase_factor = 1.12 if expected_moves <= 18 else 1.0
        tactical_factor = 1.0 + min(0.45, signals.urgency * 0.06)
        # Balanced is deliberately adaptive rather than uniformly slower: in
        # quiet positions its extra nodes were not earning their keep. It only
        # becomes more conservative when forcing root features warrant it.
        if self.profile == "very_fast":
            profile_factor = 0.50
        elif self.profile == "fast":
            profile_factor = 0.72
        elif self.profile == "balanced":
            profile_factor = 1.0 if signals.urgency >= 3 else 0.80
        else:
            profile_factor = 1.22
        soft = int(base * phase_factor * tactical_factor * profile_factor * self.time_scale)
        soft = max(self.minimum_ms, min(self.maximum_soft_ms, soft, spendable))
        # A hard limit leaves room for an unstable principal variation to be
        # verified, but it is always below the remaining game clock.
        hard = min(spendable, max(soft, int(soft * 1.55)))
        selectivity = self.selectivity_for(remaining, signals.urgency)
        if self.profile == "very_fast" and signals.urgency < 4:
            selectivity = SearchSelectivity(3, 2, 2, 3, 2, 75, 85, 6)
        elif self.profile == "fast" and signals.urgency < 4:
            selectivity = SearchSelectivity(3, 3, 2, 3, 2, 95, 105, 7)
        elif self.profile == "balanced" and signals.urgency < 3:
            selectivity = SearchSelectivity(3, 3, 2, 3, 2, 100, 110, 7)
        elif self.profile == "safe":
            selectivity = SearchSelectivity(4, 7, 1, 4, 2, 165, 170, 10)
        if self.selectivity_override == "aggressive":
            selectivity = SearchSelectivity(3, 2, 2, 3, 2, 75, 85, 6)
        elif self.selectivity_override == "safe":
            selectivity = SearchSelectivity(4, 8, 1, 4, 2, 175, 180, 10)
        return TimeBudget(soft, hard, signals, self._apply_selectivity_overrides(selectivity))

    @staticmethod
    def should_stop_after_iteration(
        elapsed_ms: int,
        budget: TimeBudget,
        stable_iterations: int,
        score_gap: int,
        mate_detected: bool,
    ) -> bool:
        """Stop at the soft limit only when the root decision is settled."""
        if elapsed_ms >= budget.hard_ms:
            return True
        if elapsed_ms < budget.soft_ms:
            return False
        if mate_detected:
            return False
        return stable_iterations >= 2 and score_gap >= 35
