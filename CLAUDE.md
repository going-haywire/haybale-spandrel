# haybale-spandrel — orientation for Claude

This repo is a **haybale library** (a plugin) for the **haywire** visual-programming
framework. It wraps [spandrel](https://github.com/chaiNNer-org/spandrel) as haywire types,
nodes and macros for upscaling and restoring images with PyTorch models. It is *not* the
framework itself — the framework lives in a separate monorepo.

## Develop from the monorepo, not from here

The framework monorepo sits next to this repo and **symlinks this repo's package into
its `barn/`**, so the same source files are editable from both sides:

```
haywire/
├── haywire-repo/                    ← framework monorepo (docs, .codemap, .insights, skills)
│   └── barn/haybale-spandrel  →  symlink to ↓
└── haybale-spandrel/                ← THIS repo (own git, .venv, releases)
    └── barn/haybale-spandrel/haybale_spandrel/   ← the actual source
```

**Prefer opening Claude sessions in `../haywire-repo`** and editing this library through
`barn/haybale-spandrel/...`. There you get the framework docs, the `.codemap/` index, the
`.insights/` traps, the `haywire-*` skills, and the live framework to import against. Edits
there land in *this* repo's files via the symlink; do git operations (commit, tag, release)
from here.

If you are working in this repo directly (offline, packaging, or a fresh clone), the
authoritative framework knowledge is at:

- `../haywire-repo/docs/` — component authoring guides + architecture
- `../haywire-repo/.codemap/INDEX.md` — repo layout map
- `../haywire-repo/.insights/` — non-obvious bugs and framework gotchas
- `../haywire-repo/CLAUDE.md` — framework conventions (reactive props, DI, testing rules)

## This repo

- Package: `haybale_spandrel` under `barn/haybale-spandrel/` (name `haybale-spandrel-dev` in dev).
- `marketstall.toml` — the published library feed manifest; updated by `haywire share`.
- Published to PyPI by `.github/workflows/publish.yml` on a `v*.*.*` tag. It needs
  `haybale-image` and `haybale-models` on PyPI first.
- Setup, running, and sharing commands are in [README.md](README.md) — don't duplicate them here.
