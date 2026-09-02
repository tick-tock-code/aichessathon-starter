"""Optional book, tablebase, persistent-state and safe pondering helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import cast

import chess


def _shipped_path(relative_path: str) -> Path | None:
    root = Path(__file__).resolve().parent
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        return None
    return path


def position_key(board: chess.Board) -> str:
    """Return the FIDE-relevant position fields used for repetition tracking."""
    return board.fen(en_passant="legal").rsplit(" ", maxsplit=2)[0]


@dataclass(slots=True)
class PositionHistory:
    """Position repetition state retained across calls in one game process."""

    counts: dict[str, int] = field(default_factory=dict)
    latest_key: str | None = None

    def reset(self) -> None:
        self.counts.clear()
        self.latest_key = None

    def observe(self, board: chess.Board) -> int:
        """Record a position once and return its observed count.

        Repeated calls with the same FEN do not double-count the current state,
        which makes it safe to call at the start of a move and during retries.
        """
        key = position_key(board)
        if key == self.latest_key:
            return self.counts.get(key, 0)
        self.latest_key = key
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def count(self, board: chess.Board) -> int:
        return self.counts.get(position_key(board), 0)

    def is_threefold(self, board: chess.Board) -> bool:
        return self.count(board) >= 3


@dataclass(slots=True)
class LookupStats:
    """Small diagnostics retained in memory; never emitted during a rated game."""

    probes: int = 0
    hits: int = 0
    misses: int = 0


class PolyglotBook:
    """Read a team-created, shipped Polyglot opening book when it is present."""

    def __init__(self, relative_path: str = "weights/opening.bin") -> None:
        self._reader: object | None = None
        self.stats = LookupStats()
        self._cache: dict[str, chess.Move | None] = {}
        path = _shipped_path(relative_path)
        if path is None or not path.is_file():
            return
        try:
            import chess.polyglot

            self._reader = chess.polyglot.open_reader(str(path))
        except (OSError, ValueError):
            self._reader = None

    @property
    def enabled(self) -> bool:
        return self._reader is not None

    def choose(self, board: chess.Board) -> chess.Move | None:
        """Choose the highest-weight legal entry deterministically, if any."""
        key = position_key(board)
        cached = self._cache.get(key)
        if cached is not None or key in self._cache:
            return cached
        self.stats.probes += 1
        if self._reader is None:
            self.stats.misses += 1
            self._remember(key, None)
            return None
        try:
            entries = list(self._reader.find_all(board))  # type: ignore[attr-defined]
            legal_entries = [entry for entry in entries if entry.move in board.legal_moves]
            if not legal_entries:
                self.stats.misses += 1
                self._remember(key, None)
                return None
            selected = max(legal_entries, key=lambda entry: (entry.weight, entry.move.uci()))
            move = cast(chess.Move, selected.move)
            self.stats.hits += 1
            self._remember(key, move)
            return move
        except (IndexError, OSError, ValueError):
            self.stats.misses += 1
            self._remember(key, None)
            return None

    def _remember(self, key: str, move: chess.Move | None) -> None:
        if len(self._cache) >= 4_096:
            self._cache.clear()
        self._cache[key] = move

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()  # type: ignore[attr-defined]
            self._reader = None
        self._cache.clear()


class SyzygyTablebases:
    """Optional local Syzygy access. Missing files always fall back to search."""

    def __init__(self, relative_path: str = "weights/syzygy") -> None:
        self._tablebase: object | None = None
        self.stats = LookupStats()
        self._cache: dict[str, chess.Move | None] = {}
        path = _shipped_path(relative_path)
        if path is None or not path.is_dir():
            return
        try:
            import chess.syzygy

            self._tablebase = chess.syzygy.open_tablebase(str(path))
        except (OSError, ValueError):
            self._tablebase = None

    @property
    def enabled(self) -> bool:
        return self._tablebase is not None

    def choose(self, board: chess.Board) -> chess.Move | None:
        """Return a WDL/DTZ-optimal legal move, or ``None`` when not covered."""
        # Halfmove clock matters to the practical 50-move-rule result, so it
        # deliberately remains part of this cache key.
        key = board.fen(en_passant="fen")
        cached = self._cache.get(key)
        if cached is not None or key in self._cache:
            return cached
        self.stats.probes += 1
        if self._tablebase is None or len(board.piece_map()) > 7:
            self.stats.misses += 1
            self._remember(key, None)
            return None
        best: tuple[int, int, str, chess.Move] | None = None
        try:
            for move in list(board.legal_moves):
                board.push(move)
                try:
                    # The child score is from the opponent's perspective.
                    wdl = -int(self._tablebase.probe_wdl(board))  # type: ignore[attr-defined]
                    try:
                        dtz = abs(int(self._tablebase.probe_dtz(board)))  # type: ignore[attr-defined]
                    except KeyError:
                        # WDL-only tablebase bundles are still useful.
                        dtz = 0
                finally:
                    board.pop()
                # Win quickly, but make an unavoidable loss take longer.
                distance_score = -dtz if wdl > 0 else dtz
                candidate = (wdl, distance_score, move.uci(), move)
                if best is None or candidate > best:
                    best = candidate
        except (KeyError, OSError, ValueError):
            self.stats.misses += 1
            self._remember(key, None)
            return None
        selected_move = None if best is None else best[3]
        if selected_move is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        self._remember(key, selected_move)
        return selected_move

    def _remember(self, key: str, move: chess.Move | None) -> None:
        if len(self._cache) >= 4_096:
            self._cache.clear()
        self._cache[key] = move

    def close(self) -> None:
        if self._tablebase is not None:
            self._tablebase.close()  # type: ignore[attr-defined]
            self._tablebase = None
        self._cache.clear()


PonderWorker = Callable[[str, Event], object | None]


@dataclass(slots=True)
class PonderResult:
    """A completed immutable prediction that search may elect to reuse."""

    predicted_fen: str
    value: object | None


class PonderController:
    """Opt-in pondering with no shared boards, TT, or search state.

    It is deliberately disabled by default.  A worker receives only an immutable
    FEN and a stop event; it must check the event regularly.  Before a timed
    search the caller must call ``stop_for_timed_search``.  That join forms the
    safety barrier: a completed result can be read, but an active worker cannot
    mutate or race with the foreground search.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._result: PonderResult | None = None

    def start(self, predicted_fen: str, worker: PonderWorker) -> bool:
        """Start one opt-in background prediction; return false when unavailable."""
        if not self.enabled:
            return False
        if not self.stop_for_timed_search():
            return False
        self._stop = Event()
        self._result = None

        def run() -> None:
            value = worker(predicted_fen, self._stop)
            if not self._stop.is_set():
                with self._lock:
                    self._result = PonderResult(predicted_fen, value)

        self._thread = Thread(target=run, name="chess-ponder", daemon=True)
        self._thread.start()
        return True

    def stop_for_timed_search(self, join_timeout_s: float = 0.02) -> bool:
        """Signal a worker and join briefly; false means the caller must not reuse it.

        The default agent keeps pondering off, so this is a defensive extension
        point rather than a background CPU consumer in normal play.
        """
        thread = self._thread
        if thread is None:
            return True
        self._stop.set()
        thread.join(timeout=max(0.0, join_timeout_s))
        if thread.is_alive():
            return False
        self._thread = None
        return True

    def take(self, current_fen: str) -> object | None:
        """Consume a completed prediction only if it matches the actual position."""
        with self._lock:
            result = self._result
            self._result = None
        if result is None or result.predicted_fen != current_fen:
            return None
        return result.value
