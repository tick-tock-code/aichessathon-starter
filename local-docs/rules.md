# Event constants

Participant-facing. Rendered by the site. Changing a row needs the organising team's sign-off.

| Constant | Value |
|---|---|
| Registration | open now, closes Sep 11 11:00 |
| Qualifier ladder | 4-11 September |
| Rated rounds | every hour, 08:00-22:00 from Sep 4 08:00 |
| Upload close | Sep 11 11:00 |
| Final qualification | 13-round Swiss over locked builds, Sep 11 afternoon |
| Finalist invites | by ~21:30 Sep 11 |
| Finalists | 48, live final Sep 12 at Encode Club, London |
| Time control | 120s plus 0.5s per move, per side, with a 60s init budget before the clock |
| Match environment | 1 CPU core, 2 GB RAM, no network, no GPU, identical hardware |
| Environment | Python 3.12 with torch, numpy, python-chess, onnxruntime and numba preinstalled at fixed versions the docs page lists. Nothing else installs and a `requirements.txt` in the zip is ignored. Ask at hello@aichessathon.com for an addition and any grant is announced to every team |
| Pondering | allowed. Your process keeps its core while the opponent thinks |
| Game end | an illegal move, a crash or a flag loses. Both sides failing voids the game. 300 plies goes to adjudication. FIDE draw rules apply |
| Openings | every game starts from a curated opening position that is close to level. Knockout ties play each position once with each colour |
| Ranking | Elo rating on the live ladder. The ladder only seeds the final Swiss |
| House bots | house engines play the ladder and show their public CCRL ratings. They cannot qualify |
| Qualification | only the locked-build final Swiss counts, by points. An odd field gives one team a 1-point bye |
| Tie-breaks | points, Buchholz, head-to-head, earlier final submission |
| Teams | 1-3 people, one team per person. Creating, joining and leaving a team close Sep 11 11:00 |
| Eligibility | Open worldwide. The final qualification Swiss and the London final are for teams where every member is a UK university student, verified before invites |
| Submissions | <= 50 MB unzipped, 6 uploads per team per day, the latest valid version plays |
| Engines | third party engines are prohibited: Stockfish, Lc0, Maia and any wrapper around one. Your moves come from code you wrote and any model you ship is one you trained. An engine you wrote yourself before the event is your own code. A model is not required, a classical search is a full entry |
| Training data | unrestricted, including positions annotated by an existing engine. The ban covers only what ships inside the submission |
| Verification | every submission faces automated and human checks. Each finalist team walks through how its agent was built. Disqualification can be retroactive |
| Books and tablebases | permitted as shipped data within the 50 MB cap. `chess.polyglot` and `chess.syzygy` are in the base image |
| Code | what you ship must be source a judge can read. Everything that runs is python from your zip plus the preinstalled stack. Obfuscated agents are disqualified |
