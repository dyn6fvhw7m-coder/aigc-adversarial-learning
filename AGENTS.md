# Project workflow

Scope: AIGC detection and adversarial-attack learning in this repository.

- Explain the purpose of a learning script, its inputs and outputs before changing or running it. The user participates in parameter choices and real experiments.
- Do not create automatic backup files; the user explicitly declined backups. Preserve unrelated changes.
- Keep datasets, pretrained weights, virtual environments, large adversarial tensors and credentials out of Git.
- Preserve third-party attribution and original license notices.
- Distinguish custom binary ensembles and APGD-T-Margin from unmodified standard AutoAttack.
- The user requested that subsequent authorized program changes also be synchronized to this GitHub repository. After relevant checks, inspect and stage the intended changes, commit with a meaningful message, and push to the configured remote. Report the commit and push result. Never force-push or overwrite remote history to resolve a conflict.
- For manual synchronization use `sync-github.ps1`; review changes before publishing. Do not describe an unsuccessful push as uploaded.
- Real experiment runs remain user-driven unless the user asks the assistant to execute them.
