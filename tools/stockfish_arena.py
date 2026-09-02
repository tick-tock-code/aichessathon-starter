"""Development-only UCI tournament runner; never include this in submission.zip.

Example:
    python tools/stockfish_arena.py --stockfish C:\\tools\\stockfish.exe --games 40
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chess

# The tool is intentionally executable directly, while the project package is
# deliberately not installed as a wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.sandbox import AgentFailure, local


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + self.draws * 0.5) / self.games if self.games else 0.0


class StockfishAgent:
    """Small adapter so the standard local referee can play a UCI engine."""

    def __init__(self, executable: Path, move_time_ms: int, skill: int | None) -> None:
        self.executable = executable
        self.move_time_ms = move_time_ms
        self.skill = skill
        self.process: subprocess.Popen[str] | None = None

    def start(self, _: float) -> None:
        try:
            self.process = subprocess.Popen(
                [str(self.executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._send("uci")
            self._wait_for("uciok")
            self._send("setoption name Threads value 1")
            self._send("setoption name Hash value 64")
            if self.skill is not None:
                self._send(f"setoption name Skill Level value {self.skill}")
            self._send("isready")
            self._wait_for("readyok")
        except OSError as error:
            raise AgentFailure("init") from error

    def move(self, fen: str, _: int) -> str:
        if self.process is None:
            raise RuntimeError("Stockfish moved before start")
        self._send(f"position fen {fen}")
        self._send(f"go movetime {self.move_time_ms}")
        while True:
            line = self._readline()
            if line.startswith("bestmove "):
                return line.split()[1]

    def stop(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self._send("quit")
                self.process.wait(timeout=2)
            self.process = None

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise AgentFailure("crash")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        if self.process is None or self.process.stdout is None:
            raise AgentFailure("crash")
        line = self.process.stdout.readline()
        if not line:
            raise AgentFailure("crash")
        return line.strip()

    def _wait_for(self, expected: str) -> None:
        while self._readline() != expected:
            pass


def _elo_difference(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def _score_interval(score: float, games: int) -> tuple[float, float]:
    """Approximate 95% interval for match score, adequate for triage only."""
    if games == 0:
        return (0.0, 1.0)
    standard_error = math.sqrt(max(1e-9, score * (1.0 - score) / games))
    return (max(0.0, score - 1.96 * standard_error), min(1.0, score + 1.96 * standard_error))


def _report(summary: ScoreSummary, reference_elo: int | None) -> None:
    score = summary.score
    interval = _score_interval(score, summary.games)
    difference = _elo_difference(score)
    lower, upper = (_elo_difference(interval[0]), _elo_difference(interval[1]))
    print(f"+{summary.wins} ={summary.draws} -{summary.losses}; score {score:.1%}")
    if difference is None:
        print("Elo difference: unbounded; run more games or use a weaker reference.")
    else:
        print(f"Elo difference vs this Stockfish setting: {difference:+.0f}")
        if lower is not None and upper is not None:
            print(f"Approximate 95% range: {lower:+.0f} to {upper:+.0f}")
        if reference_elo is not None:
            print(f"Provisional agent Elo: {reference_elo + difference:.0f}")


def run_profile(arguments: argparse.Namespace, profile: str) -> ScoreSummary:
    os.environ["CHESSATHON_TIME_PROFILE"] = profile
    wins = draws = losses = 0
    for game in range(arguments.games):
        plays_white = game % 2 == 0
        stockfish = StockfishAgent(
            arguments.stockfish, arguments.stockfish_move_ms, arguments.skill
        )
        ours = local(arguments.agent.resolve())
        white, black = (ours, stockfish) if plays_white else (stockfish, ours)
        outcome: Outcome = play_match(
            white,
            black,
            arguments.base_ms,
            arguments.increment_ms,
            arguments.ply_cap,
            arguments.start_fen,
        )
        if outcome.termination in FAILED_TERMINATIONS:
            raise SystemExit(f"agent failure in game {game + 1}: {outcome.termination}")
        if outcome.result == "draw":
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        print(
            f"{profile} game {game + 1}/{arguments.games}: "
            f"{outcome.result} ({outcome.termination})"
        )
    return ScoreSummary(wins, draws, losses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only tournament against local Stockfish."
    )
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--stockfish-move-ms", type=int, default=100)
    parser.add_argument("--skill", type=int, choices=range(0, 21))
    parser.add_argument("--reference-elo", type=int)
    parser.add_argument("--profiles", default="fast,balanced,safe")
    parser.add_argument("--ply-cap", type=int, default=300)
    parser.add_argument("--start-fen", default=chess.STARTING_FEN)
    arguments = parser.parse_args()
    if not arguments.stockfish.is_file():
        raise SystemExit(f"Stockfish executable not found: {arguments.stockfish}")
    if arguments.games <= 0 or arguments.stockfish_move_ms <= 0:
        raise SystemExit("games and stockfish-move-ms must be positive")
    for profile in tuple(item.strip() for item in arguments.profiles.split(",") if item.strip()):
        if profile not in {"fast", "balanced", "safe"}:
            raise SystemExit(f"unknown profile: {profile}")
        print(f"\n{profile} profile")
        _report(run_profile(arguments, profile), arguments.reference_elo)


if __name__ == "__main__":
    main()
