# Quickstart

Fork the starter repo: a working agent, baselines to beat, and a harness that plays games locally under the real clock. Or start from scratch, below.

## Submission layout

A submission is a zip, 50 MB unzipped at most. At its root, meaning not inside a folder:

```text
agent.zip
├── agent.py            required
└── model files         weights, books, anything else
```

The smallest legal agent:

```python
import chess
import random

def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()
```

The environment is fixed. The container ships Python 3.12, the full standard library, and five preinstalled packages at these versions:

```text
torch 2.13.0+cpu     numpy 2.5.2      python-chess 1.11.2
onnxruntime 1.29.0   numba 0.67.0
```

Nothing installs at validation and a `requirements.txt` in your zip is ignored, so importing anything outside that set crashes your agent in its smoke games. Ask at hello@aichessathon.com for a package the stack lacks and any addition is announced to every team.

Native binaries inside the zip are rejected. What you ship has to be source a judge can read. Compiled speed comes from numba, which JIT compiles your Python in process, and each jitted function pays that compile cost the first time it runs. Cython does not work here, because a compiled extension is a native binary and native binaries are rejected, and the image carries no compiler to build one at runtime. Your zip goes first on `sys.path`, so a file named after a module you import, `chess.py` or `types.py`, shadows the real one.

## Agent API

`agent.py` must expose one function:

```python
def get_move(fen: str, time_left_ms: int) -> str
```

| Name | Type | Meaning |
|---|---|---|
| `fen` | `str` | the position to move in, standard FEN |
| `time_left_ms` | `int` | your remaining clock in milliseconds |
| returns | `str` | a legal move in UCI, e.g. `e2e4` or `e7e8q` |

One process serves one game, started fresh for every game. Load your model at import or on the first call. A 60s init budget runs before the clock starts, and the process stays alive between moves, so state you keep in memory carries across your own moves. The process keeps its dedicated core after `get_move` returns, and pondering is allowed. During your own move one thread is fastest, since threads past the first share the single core and cost you time. The referee claims threefold and fifty-move draws automatically, so an agent that wants to avoid a repetition tracks the positions it has been asked about.

## Wire protocol

The runner talks to your process over stdin and stdout, one JSON object per line. You only implement `get_move`; the provided runner handles the wire.

Request:

```json
{"fen": "rnbqkbnr/...", "time_left_ms": 87500}
```

Response:

```json
{"move": "e2e4"}
```

- Your colour is the side to move in the FEN.
- `time_left_ms` is your clock before this move; the increment lands after you move.
- Output is capped at 4096 bytes per move; past it the game is lost.
- Malformed output counts as an illegal move and loses the game.
- Your own output cannot corrupt the protocol. The runner moves it to a private handle and points file descriptor 1 at stderr before importing your agent, so `print` is safe. Output is discarded during rated games and shown in the validation log, up to 8 KB.

## Match environment

| Resource | Limit |
|---|---|
| CPU | 1 dedicated core |
| Memory | 2 GB |
| Network | none, in either direction |
| GPU | none |
| Filesystem | read-only, plus 256 MB scratch at `/tmp`, where `HOME` and cache paths point |
| Processes | 128; on one core, threads past the first cost time |
| Hardware | identical for every game; both agents run on the same machine |

No network means no hosted inference and no engine APIs. Everything your agent needs ships inside the zip.

## Clocks and scoring

| Item | Rule |
|---|---|
| Time control | 120s per side, plus 0.5s per move |
| Init budget | 60s before the clock starts |
| Game end | FIDE rules; 300 plies without a result is adjudicated on material, else drawn |
| Ladder | rated rounds every hour, 08:00 to 22:00, ranked by Elo |
| Openings | every game starts from a curated position that is close to level |
| Qualification | a 13-round Swiss over locked submissions of UK university teams decides the top 48 |
| Tie-breaks | points, Buchholz, head-to-head, earlier final submission |
| Teams | 1 to 3 people, one team per person; team changes close 11 September 11:00 |
| Eligibility | ladder open worldwide; final Swiss and London final require every member to be a UK university student |
| House bots | play the ladder and show public CCRL ratings; cannot qualify |

## Submissions and permitted components

- Maximum 50 MB unzipped.
- Six uploads per team per day.
- The latest submission that passed validation plays.
- Validation builds and runs two smoke games, one as each colour; the log appears on the dashboard.
- Uploads close 11 September 11:00.
- Third-party engines are prohibited: Stockfish, Lc0, Maia, and wrappers around them.
- Moves must come from code you wrote; any shipped model must be one you trained.
- Training data is unrestricted, including positions annotated by an existing engine.
- Model weights such as `.onnx`, `.safetensors`, and `.pt` are allowed.
- Books and tablebases are allowed as shipped data; `chess.polyglot` and `chess.syzygy` are available.
- Agents must be readable source. Obfuscated agents are disqualified.

## Failure reference

| Termination | Cause | Result |
|---|---|---|
| illegal | illegal or malformed move, or output past the cap | loss |
| crash | process exited, threw, or ran out of memory | loss |
| flag | clock ran out mid-game | loss |
| init | no ready line within the 60s init budget | loss |
| adjudication | 300 plies without a result | material decides, else draw |
| void | both sides failed | no result recorded |
