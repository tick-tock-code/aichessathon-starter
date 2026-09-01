# Agent contract

The participant-facing interface, rendered at `/docs` on the site together with `RULES.md`. It does not change once the qualifier starts.

## Submission

A zip whose root holds:

- `agent.py` exposing `get_move(fen: str, time_left_ms: int) -> str` returning a UCI move (`e2e4`, `e7e8q`).
- model weights, opening books and any other files, total <= 50 MB unzipped.

At the root means `agent.py` sits at the top of the archive, not inside a folder. Most zip tools wrap the folder you selected, so check before uploading. A zip without `agent.py` at its root is rejected.

The environment is fixed. The container ships Python 3.12, the full standard library, and five preinstalled packages at the versions the docs page lists: `torch` (CPU), `numpy`, `python-chess`, `onnxruntime` and `numba`. Nothing installs at validation and a `requirements.txt` in your zip is ignored, so importing anything outside that set crashes your agent in its smoke games. Ask at hello@aichessathon.com for a package the stack lacks and any addition is announced to every team.

Native binaries inside the zip are rejected. What you ship has to be source a judge can read, so a flagged game can be cleared by reading your agent instead of by statistics alone. Model weights are not binaries, so `.onnx`, `.safetensors` and `.pt` are fine. Any model you ship is one you trained yourself. Your zip is first on `sys.path`, so a file named after a module you import, like `chess.py` or `types.py`, shadows the real one.

Compiled speed comes from the stack, not from your zip. numba JIT compiles your Python in process, and each jitted function pays that compile cost the first time it runs. Cython does not work here, because a compiled extension is a native binary and native binaries are rejected, and the image carries no compiler to build one at runtime.

## Execution

One persistent process per agent per game, started fresh for every game. Load your model at import or first call. A 60s init budget runs before the clock starts. Between moves the process stays alive, so state you keep in memory carries across your own moves. It does not carry to your next game. The process keeps its dedicated core after `get_move` returns, and pondering is allowed. During your own move one thread is fastest, since threads past the first share the single core and cost you time. Games are independent, so two of your games can run at the same time in separate containers. Your agent is never asked for two moves at once. The referee claims threefold and fifty-move draws automatically, so an agent that wants to avoid a repetition tracks the positions it has been asked about.

The runner handles the wire. It writes one JSON line to your process per move request:

```json
{"fen": "...", "time_left_ms": 87500}
```

and reads one JSON line back:

```json
{"move": "e2e4"}
```

`time_left_ms` is your remaining clock before this move. The increment lands after you move. Your colour is the side to move in the fen. The first fen you receive is the starting position of the game. Games start from curated opening positions, not always the standard start. Every starting position is close to level. The set of positions is not published in advance, though finished games reveal the positions they were played from. Repetition and fifty-move counts begin there.

Your own output cannot corrupt this. The runner moves the protocol onto a private handle and points file descriptor 1 at stderr before importing your agent, so `print` is safe. Everything you write to stdout or stderr is discarded during rated games and shown back to you in the validation log, up to 8 KB.

## Match conditions

- 1 dedicated CPU core, 2 GB RAM, no network, no GPU. Identical hardware for all games. Both agents of a game run on the same machine.
- The filesystem is read-only apart from 256 MB at `/tmp`. `HOME`, `TORCH_HOME`, `HF_HOME` and the other cache paths already point there. `/tmp` starts empty for every game and is deleted with the game, so use it as scratch space, not as a cache between games.
- 128 processes, one core. Threads past the first cost you time rather than winning it.
- The clock is 120s base plus 0.5s per move, per side, enforced on wall time by the runner.
- 300 plies without a result goes to material adjudication, else a draw. Draws otherwise follow FIDE rules (python-chess).

## Failure semantics

An illegal move, malformed output, a move payload over 4 KB, a crash, running out of memory, or a flag fall loses that game. If both sides fail the game is void. There are no retries within a game.

Validation plays two smoke games against a house agent, one as each colour, and publishes the verbatim log. Both games start from curated opening positions.

## Versions

The latest submission that passed validation plays each round. Uploads close Sep 11 11:00. Your last valid build then freezes, and for eligible teams it alone plays the final qualification Swiss, which does not open until every submission has finished validating. Six uploads per team per day.
