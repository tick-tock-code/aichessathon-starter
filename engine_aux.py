"""Optional book, tablebase, persistent-state and safe pondering helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


class PgnEndgameTablebase:
    """Read exact endgame choices exported as annotated Syzygy PGNs.

    This is intentionally separate from :class:`SyzygyTablebases`: the latter
    consumes native ``.rtbw``/``.rtbz`` files, while this compact fallback
    consumes a team-supplied PGN export with ``WDL`` and move-list headers.
    """

    def __init__(self, relative_path: str = "weights/endgames.pgn", max_pieces: int = 3) -> None:
        self.max_pieces = max(2, max_pieces)
        self.stats = LookupStats()
        self._moves: dict[str, tuple[chess.Move, ...]] = {}
        path = _shipped_path(relative_path)
        if path is None or not path.is_file():
            return
        try:
            import chess.pgn

            with path.open(encoding="utf-8", errors="replace") as handle:
                while (headers := chess.pgn.read_headers(handle)) is not None:
                    self._load_headers(headers)
        except (OSError, ValueError):
            self._moves.clear()

    @property
    def enabled(self) -> bool:
        return bool(self._moves)

    def choose(self, board: chess.Board) -> chess.Move | None:
        """Return an annotated legal move for a covered endgame position."""
        self.stats.probes += 1
        moves = self._moves.get(position_key(board))
        if not moves:
            self.stats.misses += 1
            return None
        legal = [move for move in moves if move in board.legal_moves]
        if not legal:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return min(legal, key=chess.Move.uci)

    def _load_headers(self, headers: object) -> None:
        # Headers is intentionally treated through ``get`` so the loader has
        # no runtime dependency on python-chess's private PGN header types.
        get = getattr(headers, "get", None)
        if not callable(get):
            return
        fen = get("FEN")
        wdl = get("WDL")
        if not isinstance(fen, str) or not isinstance(wdl, str):
            return
        try:
            board = chess.Board(fen)
        except ValueError:
            return
        if len(board.piece_map()) > self.max_pieces:
            return
        header_name = {"Win": "WinningMoves", "Draw": "DrawingMoves", "Loss": "LosingMoves"}.get(
            wdl
        )
        if header_name is None:
            return
        raw_moves = get(header_name, "")
        if not isinstance(raw_moves, str):
            return
        moves: list[chess.Move] = []
        for san in raw_moves.split(","):
            try:
                moves.append(board.parse_san(san.strip()))
            except ValueError:
                continue
        if moves:
            self._moves[position_key(board)] = tuple(moves)


PonderWorker = Callable[[str, Event], object | None]


@dataclass(slots=True)
class PonderResult:
    """A completed immutable prediction that search may elect to reuse."""

    predicted_fen: str
    value: object | None


@dataclass(slots=True)
class PonderStats:
    """In-memory observability for local ponder experiments."""

    branches_started: int = 0
    branches_completed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cancellations: int = 0


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
        self._results: dict[str, PonderResult] = {}
        self.stats = PonderStats()

    def start(self, predicted_fen: str, worker: PonderWorker) -> bool:
        """Start one opt-in background prediction; retained as a simple API."""
        return self.start_many((predicted_fen,), worker)

    def start_many(
        self,
        predicted_fens: Sequence[str],
        worker: PonderWorker,
        *,
        max_branches: int = 3,
    ) -> bool:
        """Prepare a small ordered portfolio of replies on one worker thread.

        Branches are deliberately searched sequentially. The competition grants
        one CPU core, so parallel branches would compete with each other rather
        than make useful preparation faster.
        """
        if not self.enabled:
            return False
        if not self.stop_for_timed_search():
            return False
        limit = max(1, max_branches)
        unique_fens = tuple(dict.fromkeys(predicted_fens))[:limit]
        if not unique_fens:
            return False
        self._stop = Event()
        with self._lock:
            self._results.clear()

        def run() -> None:
            for predicted_fen in unique_fens:
                if self._stop.is_set():
                    return
                self.stats.branches_started += 1
                value = worker(predicted_fen, self._stop)
                if self._stop.is_set():
                    return
                with self._lock:
                    self._results[predicted_fen] = PonderResult(predicted_fen, value)
                    self.stats.branches_completed += 1

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
            self.stats.cancellations += 1
            return False
        self._thread = None
        return True

    def take(self, current_fen: str) -> object | None:
        """Consume a completed prediction only if it matches the actual position."""
        with self._lock:
            result = self._results.get(current_fen)
            self._results.clear()
        if result is None:
            self.stats.cache_misses += 1
            return None
        self.stats.cache_hits += 1
        return result.value
