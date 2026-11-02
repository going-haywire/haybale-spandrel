# Upscale Image

`haybale-spandrel:node:UpscaleImageNode` · kind: node

Enlarges or restores an image with a loaded model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `image` | inlet | `haybale-image:type:IMAGE` | A picture of any channel layout. Pixels are kept as made and converted on demand. |
| `model` | inlet | `haybale-spandrel:type:MODEL` | A model loaded on a device, ready to run. Never saved with the graph. |
| `scale` | inlet | `haywire-core:type:FLOAT` | 0 = the model's own scale |
| `tile_size` | config | `haywire-core:type:STRING` | Text data |
| `alpha` | config | `haywire-core:type:STRING` | Text data |
| `progress.main` | config | `haywire-core:type:PROGRESS` |  |
| `out_image` | outlet | `haybale-image:type:IMAGE` | A picture of any channel layout. Pixels are kept as made and converted on demand. |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `main` | progress | `PROGRESS(active=False, fraction=None, label='')` |  |

## Notes

Enlarges or restores an image with a loaded model.

Connect `Model` to Load Model. The result stays on the model's device, so a
following model step runs without copying.

- `Scale`: size of the result relative to the image; `0` uses the model's own
  scale. A larger scale runs the model again (a 4x model twice for 8x or 16x);
  a scale between steps is reached by scaling the result down smoothly.
- `Tile Size`: large images run in tiles that fit the memory budget (*Spandrel*
  settings). `auto` picks the size, `none` runs the whole image at once, a number
  sets the tile side in pixels. A model that tiles itself is never tiled.
- `Alpha`: for images with transparency. `resize` scales the transparency
  smoothly; `model` runs it through the model too, which takes twice as long and
  keeps edges sharper.

A gray image comes back gray. A colour image through a grayscale model comes
back gray, with a note on the card.
