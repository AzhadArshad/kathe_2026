"""Learned character-level diacritic restoration (R11).

`data.diacritize` holds the lexicon restorer (submissions 003-007). This package
holds the neural one. Both restore exactly the same three marks and share
`data.diacritize.RESTORABLE` / `strip_key` so there is one definition of what
"restorable" means.
"""
