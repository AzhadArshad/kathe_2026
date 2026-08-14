# Submission 010 — per-mark calibrated thresholds

**LEADERBOARD 10.64.** Worse than sub 009 (11.78) and sub 007 (11.81).

Changed ONE thing from 009: the mark mix, calibrated by binary search so each
mark's density hit R0's reference share.

| mark | sub 009 | sub 010 | R0 reference |
| --- | ---: | ---: | ---: |
| kasra | 1.94 | 1.96 | 2.31 |
| damma | 1.83 | 1.13 | 1.36 |
| fatha | 0.29 | 0.83 | 1.00 |

Total density stayed in the validated band (9.36 against 009's 9.71 and 007's
9.32). It still lost 1.14 points.

**Two things this establishes:**

1. **The test references do not carry R0's per-mark profile.** R0's fatha rate
   is an artifact of its `daily`-heavy sampling, like its overall density.
2. **Density is necessary but not sufficient.** 007 at 9.32 scores 11.81; this
   at 9.36 scores 10.64.

Do not re-derive this configuration. Fitting output statistics to R0 has now
lost points three times (005, 008, 010).
