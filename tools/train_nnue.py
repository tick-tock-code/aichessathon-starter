"""Train a compact team-owned NNUE-style evaluator from Lichess PGNs.

This is development-only.  It never ships in ``submission.zip`` and it never
invokes an engine at agent runtime.  The initial outcome-labelled pass is a
quick smoke model; a later annotation pass can replace labels with offline
Stockfish centipawn targets without changing the exported model format.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_neural import DEFAULT_HIDDEN_SIZE, FEATURE_COUNT  # noqa: E402

MAX_PIECES = 32
PADDING_FEATURE = FEATURE_COUNT


def _feature_index(piece: chess.Piece, square: chess.Square) -> int:
    colour_offset = 6 if piece.color == chess.WHITE else 0
    return (colour_offset + piece.piece_type - 1) * 64 + square


def _outcome_target(result: str, scale: float) -> float | None:
    if result == "1-0":
        return scale
    if result == "0-1":
        return -scale
    if result == "1/2-1/2":
        return 0.0
    return None


def sample_outcome_positions(
    pgn_path: Path, max_games: int, seed: int, outcome_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Take one deterministic middlegame position per completed game."""
    random_source = random.Random(seed)
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        games_seen = 0
        while games_seen < max_games:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games_seen += 1
            target = _outcome_target(game.headers.get("Result", ""), outcome_scale)
            moves = list(game.mainline_moves())
            if target is None or len(moves) < 16:
                continue
            # Avoid book and terminal positions while maintaining reproducibility.
            chosen_ply = random_source.randrange(8, min(len(moves) - 1, 80))
            board = game.board()
            for move in moves[:chosen_ply]:
                board.push(move)
            if board.is_game_over(claim_draw=True):
                continue
            row = np.full(MAX_PIECES, PADDING_FEATURE, dtype=np.int64)
            for index, (square, piece) in enumerate(board.piece_map().items()):
                row[index] = _feature_index(piece, square)
            feature_rows.append(row)
            labels.append(target)
    if not feature_rows:
        raise SystemExit("no usable completed games were found in the PGN")
    return np.stack(feature_rows), np.asarray(labels, dtype=np.float32)


class OutcomeNNUE(nn.Module):
    """Sparse feature-sum model matching the runtime accumulator layout."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(FEATURE_COUNT + 1, hidden_size, padding_idx=PADDING_FEATURE)
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.03)
        with torch.no_grad():
            self.embedding.weight[PADDING_FEATURE].zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        summed = self.embedding(features).sum(dim=1) + self.hidden_bias
        # Runtime activations are quantised and clipped to [0, 127].  Training
        # in this range makes the exported evaluator faithful to this module.
        hidden = torch.clamp(summed, 0.0, 2.0)
        return self.output(hidden).squeeze(1)


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> OutcomeNNUE:
    torch.manual_seed(seed)
    indices = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.9))
    train_indices, validation_indices = indices[:split], indices[split:]
    train_data = TensorDataset(
        torch.from_numpy(features[train_indices]), torch.from_numpy(labels[train_indices])
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = OutcomeNNUE(hidden_size)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.HuberLoss(delta=120.0)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(batch_labels)
        model.eval()
        with torch.no_grad():
            if len(validation_indices):
                valid_features = torch.from_numpy(features[validation_indices])
                valid_labels = torch.from_numpy(labels[validation_indices])
                validation_loss = float(loss_fn(model(valid_features), valid_labels))
            else:
                validation_loss = 0.0
        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_huber={total_loss / len(train_data):.2f} valid_huber={validation_loss:.2f}"
        )
    return model


def export_quantised(model: OutcomeNNUE, output_path: Path) -> None:
    """Export the integer arrays consumed by :class:`engine_neural.SparseNNUE`."""
    activation_scale = 64.0
    output_scale = 256
    with torch.no_grad():
        feature_weights = torch.clamp(
            torch.round(model.embedding.weight[:FEATURE_COUNT] * activation_scale), -32768, 32767
        ).to(torch.int16).cpu().numpy()
        hidden_bias = torch.clamp(
            torch.round(model.hidden_bias * activation_scale), -(2**31), 2**31 - 1
        ).to(torch.int32).cpu().numpy()
        output_weights = torch.clamp(
            torch.round(model.output.weight[0] * output_scale / activation_scale), -32768, 32767
        ).to(torch.int16).cpu().numpy()
        output_bias = np.asarray(
            int(torch.round(model.output.bias[0] * output_scale).item()), dtype=np.int32
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        feature_weights=feature_weights,
        hidden_bias=hidden_bias,
        output_weights=output_weights,
        output_bias=output_bias,
        output_scale=np.asarray(output_scale, dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export a compact NNUE candidate")
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=50_000)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--outcome-scale", type=float, default=350.0)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    if not arguments.pgn.is_file():
        raise SystemExit(f"PGN not found: {arguments.pgn}")
    if arguments.max_games <= 0 or arguments.hidden_size <= 0:
        raise SystemExit("max-games and hidden-size must be positive")
    features, labels = sample_outcome_positions(
        arguments.pgn, arguments.max_games, arguments.seed, arguments.outcome_scale
    )
    print(f"sampled {len(labels)} positions from {arguments.pgn.name}")
    model = train_model(
        features,
        labels,
        arguments.hidden_size,
        arguments.epochs,
        arguments.batch_size,
        arguments.seed,
    )
    export_quantised(model, arguments.output)
    print(f"wrote {arguments.output} ({arguments.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
