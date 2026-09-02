# Engine implementation status

This branch implements the twelve proposed engine ideas as a modular research platform. The
default competition mode is the stable alpha-beta engine. Experimental modes are selectable for
local matches, while learned components activate only when team-created assets are shipped.

## Composition

| Ideas | Implementation | Default status |
|---|---|---|
| 1–5 | Iterative-deepening alpha-beta/PVS, TT, ordering, quiescence, tapered evaluation, selective search, packed Numba evaluation | Active |
| 6 | Sparse quantized NNUE-style evaluator with incremental accumulators | Activates with `weights/nnue.npz` |
| 7 | Polyglot opening-book and Syzygy tablebase lookup | Activates with shipped files |
| 8 | Cached CPU ONNX policy ordering at the root | Activates with `weights/policy.onnx` |
| 9 | Predicted-reply pondering with cancellable isolated search | Opt-in with `CHESSATHON_PONDER=1` |
| 10 | Single-core PUCT with a transposition DAG | Experimental mode |
| 11 | Square-token/search-free policy interface | Experimental mode; trained inference callback still needs team weights |
| 12 | Phase mixture-of-experts, implicit action-value search, and DAG sharing | Experimental mode |

No third-party engine code or weights are included. Neural strength, opening knowledge, and
tablebase coverage require assets produced or selected by the team in compliance with the live
rules.

## Source map

- `agent.py` composes the engine and is the competition entry point.
- `engine_core.py` contains the stable alpha-beta search.
- `engine_eval.py` contains the hand-crafted and Numba-compiled evaluation.
- `engine_neural.py` contains NNUE-style and ONNX policy components.
- `engine_aux.py` contains book, tablebase, history, and pondering helpers.
- `engine_experimental.py` contains MCTS/PUCT, DAG, square-token, MoE, and implicit-search paths.
- `tests/test_engines.py` contains focused correctness and integration regressions.

The harness remains unmodified.

## Selecting an experimental engine locally

Set `CHESSATHON_ENGINE` before starting the agent process:

| Value | Search path |
|---|---|
| `alphabeta` | Stable default |
| `mcts` | PUCT using the square-token fallback policy and phase-expert value |
| `dag` | Alias emphasizing the MCTS transposition-DAG implementation |
| `moe` | Alias emphasizing the phase-expert value route |
| `policy` | Search-free square-token policy |
| `implicit` | Action-value/implicit-search selector |

Unknown values safely fall back to `alphabeta`.

## Optional shipped assets

The loaders are fail-closed: a missing, invalid, or incompatible asset disables that component.

- `weights/nnue.npz`: arrays `feature_weights[768, hidden]`, `hidden_bias[hidden]`,
  `output_weights[hidden]`, scalar `output_bias`, and optional scalar `output_scale`.
- `weights/policy.onnx`: CPU ONNX model accepting `board` with shape `[1, 12, 8, 8]` and
  producing at least 20,480 logits using the module's move-index convention.
- `weights/opening.bin`: Polyglot opening book.
- `weights/syzygy/`: selected Syzygy files.

The package builder includes root Python modules and the `weights/` directory, but not this
documentation or the tests.

## Verification

Run:

```text
pixi run lint
pixi run typecheck
pixi run test
pixi run smoke
pixi run zip
```

Strength claims require paired matches over many games. The focused tests prove legality,
terminal handling, model-accumulator consistency, experimental-mode safety, and packaging—not
final Elo.
