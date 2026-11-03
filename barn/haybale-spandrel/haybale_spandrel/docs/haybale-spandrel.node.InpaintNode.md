# Inpaint

`haybale-spandrel:node:InpaintNode` · kind: node

Fills the masked part of an image with a loaded inpainting model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `image` | inlet | `haybale-image:type:IMAGE` | Image to fill |
| `mask` | inlet | `haybale-image:type:IMAGE_GRAY` | White marks what to fill |
| `model` | inlet | `haybale-spandrel:type:INPAINT_MODEL` | A loaded inpainting model |
| `out_image` | outlet | `haybale-image:type:IMAGE` | The filled image |

## Notes

Fills the masked part of an image with a loaded inpainting model, such as LaMa.

Connect `Model` to Load Spandrel Model. White in `Mask` marks what to fill; a mask of a
different size is stretched to the image. The whole image runs at once, and
transparency is kept as it was. A super-resolution model here is an error.
