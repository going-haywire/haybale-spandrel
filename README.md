# Haybale Spandrel

Upscale and restore images in [haywire](https://github.com/going-haywire/haywire) graphs with PyTorch models, using [spandrel](https://github.com/chaiNNer-org/spandrel).

## Subscribe

<!-- marketstall:share-url:start -->
Always the latest (tracks the current branch):

```sh
https://github.com/going-haywire/haybale-spandrel/blob/master/marketstall.toml
```

Frozen to this version:

```sh
https://github.com/going-haywire/haybale-spandrel/blob/v0.0.4/marketstall.toml
```
<!-- marketstall:share-url:end -->

## Getting started

```sh
uv sync
uv run haywire
```

Development links the framework through `../haywire-repo`, so that checkout has to sit next to this one.

## Share the library

Run from the project root (this repo), on `master` with a clean tree. Run `git checkout uv.lock` first (a plain `uv run` re-locks and dirties it), and use `--frozen`:

```sh
uv run --frozen haywire share --dry-run                      # report only, writes nothing
uv run --frozen haywire share --bump patch|minor|major
```

`share` bumps the version, regenerates the docs and `marketstall.toml`, then commits, tags `v<version>` and pushes itself. The tag starts `.github/workflows/publish.yml`, which builds the wheel and publishes it to PyPI.

With `distribute = "pypi"` the first release has to be started by hand via `workflow_dispatch` of `publish.yml`; see "The first release" in `../haywire-repo/docs/guides/publish-to-pypi.md` (draft PR #18).
