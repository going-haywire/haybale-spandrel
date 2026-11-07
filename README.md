# Haybale Spandrel

Upscale and restore images in [haywire](https://github.com/going-haywire/haywire) graphs with PyTorch models, using [spandrel](https://github.com/chaiNNer-org/spandrel).

## Subscribe

<!-- marketstall:share-url:start -->
```sh
*Subscribe URL not yet published — run `haywire share --save` after pushing this repo to a git remote.*
```
<!-- marketstall:share-url:end -->

## Getting started

```sh
uv sync
uv run haywire
```

Development links the framework through `../haywire-repo`, so that checkout has to sit next to this one.

## Share the library

This bumps the version, regenerates the docs and `marketstall.toml`, commits, tags and pushes. The tag starts `.github/workflows/publish.yml`, which builds the wheel and publishes it to PyPI.

```sh
uv run haywire share --bump patch|minor|major
```
