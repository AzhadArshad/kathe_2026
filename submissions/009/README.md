# Submission 009 — R11 learned restorer, none_bias -1.0

**LEADERBOARD 11.78**, against sub 007's 11.81 — a 0.03 gap, i.e. a tie.

Identical to 008 except `restore_none_bias: -1.0`, chosen to put output density
on the reference (9.71 against 9.76) rather than at the R0 optimum. That single
knob is worth **+1.73 leaderboard points** over 008's 10.05.

**Verdict: the learned restorer matches the lexicon and does not beat it.** Sub
007 remains production — it is smaller, needs no torch at inference, and scores
the same.
