# Haybale Spandrel

A haywire project.

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
## Share your library

this will bump all the files to the specified version, commit, and sets the git-tag
then it generates and saves the marketstall.toml file and updates the above
subscribe link with the url to get this library. The version can be an explicit
`x.y.z` or an npm-style `major`/`minor`/`patch` keyword.
```sh
uv run haywire share --bump patch --save
```
