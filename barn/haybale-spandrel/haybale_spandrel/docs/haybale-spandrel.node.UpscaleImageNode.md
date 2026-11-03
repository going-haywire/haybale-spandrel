# Upscale Image

`haybale-spandrel:node:UpscaleImageNode` · kind: node

Enlarges or restores an image with a loaded model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `image` | inlet | `haybale-image:type:IMAGE` | Image to process |
| `model` | inlet | `haybale-spandrel:type:IMAGE_MODEL` | A loaded image-to-image model |
| `scale` | inlet | `haywire-core:type:FLOAT` | Size of the result relative to the input; 0 is the model's own scale |
| `tile_size` | config | `haywire-core:type:STRING` | Tile edge in pixels; auto fits the memory, none runs the whole image at once |
| `alpha` | config | `haywire-core:type:STRING` | Alpha channel: resize scales it smoothly, model runs it through the model |
| `progress.main` | config | `haywire-core:type:PROGRESS` |  |
| `out_image` | outlet | `haybale-image:type:IMAGE` | The processed image |

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
