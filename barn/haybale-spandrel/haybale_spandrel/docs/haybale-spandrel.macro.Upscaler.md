# Upscaler

`haybale-spandrel:macro:Upscaler` · kind: macro

## Ports

| id | direction | type | description |
|---|---|---|---|
| `in_slot_0` | inlet | `haybale-models:type:MODEL_REF` | Model to load: a catalog entry or a file |
| `in_execute` | inlet | `haywire-core:type:EXEC` | Lists the folder's files again |
| `in_slot_2` | inlet | `haywire-core:type:STRING` | File names to match, such as \*.png<br><br>Separate patterns with ; as in \*.png;\*.jpg<br>\* any characters, <br>? one character, <br>[abc] one of a, b or c<br>Upper and lower case match alike; <br>empty matches every file |
| `in_slot_1` | inlet | `haywire-core:type:FOLDER_PATH` | Folder to list |
| `in_slot_3` | inlet | `haywire-core:type:FOLDER_PATH` | Folder the file goes in |
| `in_slot_4` | inlet | `haywire-core:type:STRING` | File extension, png or jpg or tiff |
| `out_next` | outlet | `haywire-core:type:EXEC` | Runs once the folder exists |
| `out_out_image` | outlet | `haybale-image:type:IMAGE` | The processed image |
| `out_path` | outlet | `haywire-core:type:FILE_PATH` | The composed file path |
