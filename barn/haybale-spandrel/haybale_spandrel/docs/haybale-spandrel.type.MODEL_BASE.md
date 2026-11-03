# Loaded model

`haybale-spandrel:type:MODEL_BASE` · kind: type

A model loaded on a device, ready to run. Never saved with the graph.

## Details

- **wire_type**: `haywire-core:wire:DataWire`
- **default**: `{'value': {}}`
- **color**: `#d97ab0`

## Notes

A model loaded on a device in a precision, ready to run, of any kind.

Spandrel Model Info takes any loaded model through it. Upscale Image and Inpaint take
one kind each (`IMAGE_MODEL`, `INPAINT_MODEL`), so a Load Spandrel Model wired to them
becomes that kind and offers only the models that fit.
