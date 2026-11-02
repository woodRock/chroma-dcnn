# Rules for agents in this repo (taura provenance)

- Never type a numeric result into any .tex file. Add it to a result in
  ledger/results/ and reference it with \res{key}.
- Never edit anything in the paper's generated/ folder. Regenerate it.
- Never copy a figure into the paper by hand. Add a figure artifact in
  ledger/artifacts.toml and publish it (`taura publish`, or
  `taura publish --no-commit` mid-revision — see below).
- Launch experiments with `taura run -- <cmd>` (or via gpu-spooler with
  taura recording enabled) so every run gets a record.
- Before saying a task is done, run `taura check`. If it reports
  anything, report the findings to the user. Do not "fix" a hand-edit
  finding by overwriting the file; a collaborator may have made it.
- Use `taura diff` to describe what a change does to the paper.
- Never `git push` the paper repo.

## Mid-revision: syncing into a deliberately dirty paper repo

`taura publish` refuses outright whenever the paper repo has uncommitted
changes — there's no override flag, because publish always commits there.
That's the common case while actively rewriting the paper (e.g. the author
wants an uncommitted working tree so Overleaf/latexdiff can show a clean
diff of their own edits to send to collaborators), and it blocks every
figure/result rerun until they commit, which they may not want to do yet.

Use `taura publish --no-commit` instead: it copies the built artifacts into
the paper's `generated/` and rewrites `taura.lock` to match (so a later
`taura check` doesn't mistake the sync for a hand edit) but leaves both
uncommitted — the paper repo's git state is untouched. `taura status` keeps
showing it as unpublished. A later plain `taura publish`, once the paper
repo is clean again, commits the already-matching lock for real (a no-op if
nothing else changed in the meantime).

## Reruns on the ECS GPU cluster

`scripts/rerun_pipeline_on_ecs.sh [HOST] [--skip-download]` chains the full
preprocess → pretrain → fine-tune/eval → saliency pipeline as one GPU job via
`task submit`/`task wait`/`task fetch` (see `task --help`), re-downloading
the pretraining corpus first if it's missing on the server. Needed whenever
a revision changes code, configs, or data that the reported numbers or
figures depend on. Run it, then `taura check` and `taura publish
[--no-commit]`.

Two things that aren't obvious from `task`'s own `--help`:
- `task exec`'s SSH session has its own ~300s client-side timeout; a command
  that outlives it (e.g. the MassBank download, 15-30 min over the shared
  volume) keeps running detached on the server regardless — check with
  `task exec -H HOST -- "pgrep -fa <script name>"` rather than assuming it
  died. `task submit` (used for the GPU steps) doesn't have this problem;
  it's built for long-running jobs from the start.
- Remote paths follow `~/.gpu-spooler/hosts.toml`'s `[groups.ecs]
  remote_root` + this repo's name, not the repo's local path.

## Moving typed numbers into \res (taura adopt)

1. `taura adopt --todo --json` lists the typed numbers that need a decision,
   each with an `id`, the sentence and paragraph around it, and the
   candidate result keys.
2. Read the sentence. Decide what the number is and record it:
   - `taura resolve ID KEY --reason "..."` if it reports that result;
   - `taura resolve ID --none --reason "..."` if it isn't a result (year,
     setting, sample size, a value from another paper, a coincidence);
   - `taura resolve ID --unrecorded --reason "..."` if it is a computed
     result but no candidate fits (report these: the ledger needs them).
   `taura resolve --accept-unique` records every number with exactly one
   strong match. Or let an agent CLI decide: `taura adopt --agent "claude -p"`.
3. Replace each number decided as a result with `\res{KEY}` in the .tex.
   Change nothing else on the line.
4. `taura check`: `adopt-mismatch` means a key doesn't print what the paper
   said (add a [format] entry). Report anything else to the user.
