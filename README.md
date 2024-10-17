## CLIP-tokenizer

- A simple CLIP tokenizer encode / decode script. For when you need to know CLIP's tokenization and / or IDs for some reason.
---------
- `python clip-tokenizer.py --text "a photo of a cat"` - returns:
- a photo of a cat   ['a</w>', 'photo</w>', 'of</w>', 'a</w>', 'cat</w>']  320,1125,539,320,2368
---------
- `python clip-tokenizer.py --text "333, 6554, 5322" --reverse` with `--reverse`, get tokens for comma-separated IDs:
- n kabrie  ['n</w>', 'kab', 'rie</w>']  333,6554,5322
---------
- `python clip-tokenizer.py --file exampleTEXT.txt` with `--file`, batch processes file (newline = separate input).
- `python clip-tokenizer.py --file exampleIDs.txt --reverse` batch process, but for token IDs. Both save output as file.
---------
