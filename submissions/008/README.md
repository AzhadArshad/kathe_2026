# Submission 008 — R11 learned restorer, none_bias 0.0

**LEADERBOARD 10.05.** A 1.76 regression from sub 007's 11.81, and the largest
in the project.

Same decode as 002-007; the only change is replacing the diacritic lexicon with
the R11 character-level tagger at its R0-optimal bias.

**It was submitted BECAUSE the two offline signals disagreed**, and it settled
which one to trust:

| signal | said | outcome |
| --- | --- | --- |
| R0 geo | 34.02, best ever, +0.66 over sub 007 | **wrong** |
| output density | 7.65/100c against references at 9.76 — the lowest ever submitted, worse than sub 005 which regressed | **right** |

Submitted alongside 009 (the same model at bias -1.0, density-matched) precisely
so one submission pair would resolve it. It did: 009 scored 11.78.

Do not re-derive this configuration from its R0 score.
