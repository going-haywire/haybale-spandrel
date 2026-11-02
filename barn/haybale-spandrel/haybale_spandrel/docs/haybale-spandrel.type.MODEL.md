# Loaded model

`haybale-spandrel:type:MODEL` · kind: type

A model loaded on a device, ready to run. Never saved with the graph.

## Details

- **wire_type**: `haywire-core:wire:DataWire`
- **default**: `{'value': {}}`
- **color**: `#d97ab0`

## Notes

A model loaded on a device in a precision, ready to run.

Load Model makes one from a model you pick; Upscale Image, Inpaint and Model Info
take it. Several nodes can share one: the model is loaded once.
