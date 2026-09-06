# Re-validating the original heuristic at 8-region scale

[surge-exposure](https://github.com/DBishal13/surge-exposure)'s validation
study only ever checked `exposure_score` against real losses for one county
(Lee County, FL / Hurricane Ian). This project independently fetched real
NFIP claims across all 8 regions it covers — 140,732 claims, not just Lee
County's 48,105 — which makes it possible to re-run that same study's exact
methodology at a much larger geographic footprint, without touching a line
of the original scoring code.

**Method** (`revalidate_original_study.py`, a faithful replication of
`surge-exposure/scripts/validate_exposure_bins.py`): snap buildings and
claims to the same 0.1° grid, aggregate each side per cell (mean
`exposure_score` / building count vs. claim count / mean payout),
inner-join, Pearson-correlate. Only the geographic scope changes.

## Results — 13 cells, 8 regions

| Comparison | r (8 regions, cell-level) | Original (Lee County only) |
|---|---:|---:|
| `exposure_score` vs. claim count | **-0.113** | 0.37 (0.25 under the Ian-window check) |
| `exposure_score` vs. mean claim payout | **0.711** | 0.52 (0.516 under the Ian-window check) |

**The severity signal replicates — and gets stronger, not weaker — at 8x
the geographic scope.** That's the more important of the two headline
numbers, and it moves the right direction: a heuristic that only worked by
Lee-County-specific coincidence would be expected to weaken outside it, not
strengthen. Excluding one outlier cell (see below), the severity
correlation rises further, to **r = 0.805** across the remaining 12 cells —
the original study's central finding holds up, and holds up better,
outside the county it was discovered in.

**The frequency signal doesn't just weaken further, it becomes unstable.**
r = -0.113 across all 13 cells, but **r = +0.169 with one cell excluded** —
the sign itself depends on a single data point out of 13. Read this as
confirmation, not contradiction, of the original study's own Ian-window
finding (r=0.37 → 0.25): claim frequency was never a signal this heuristic
tracks reliably, and at wider scope that unreliability shows up as sign
instability rather than a merely smaller number.

## The actual finding worth having: French Quarter, New Orleans

One cell scores **`exposure_score = 0.000`** — the heuristic's flat
statement that this location has *no* storm-surge exposure at all — and
has **7,931 real NFIP claims**, the single highest claim count of any cell
in the entire 8-region dataset (more than 2x the next-highest), averaging
**$64,576** in building-only payouts per claim.

This isn't a subtle miss. It's the largest, most damaging blind spot
findable in this data, and it's exactly the failure mode
`surge-exposure`'s own README already names in the abstract ("~30% of the
county's real claims came from inland, rainfall-driven flooding a
surge-only signal was never going to see") — except here it's concrete: a
named place, a specific 0.1°×0.1° cell, a real claim count. New Orleans'
flood risk is famously levee/rainfall/riverine-driven, not surge-driven,
which is precisely the gap `flood_active`'s live-only NOAA feed (no
historical replay) can't close.

## Why this matters for the next step

This was found using nothing but the existing heuristic and real claims —
no trained model needed to locate it. But it's the first concrete instance
of exactly the kind of blind spot the strategy's next step (comparing the
trained model's predictions against the heuristic, cell by cell) is meant
to surface systematically rather than by inspection. French Quarter, NOLA
is now a specific, named test case: does `surge-exposure-ml`'s trained
severity model also fail there, or does it pick up signal the hand-picked
heuristic structurally cannot?

## Answered: yes, but not for the reason you'd want

Queried the live Model Serving endpoint for a representative French
Quarter building (`surge_ft=0, height_m=4.78, flood_active=0,
region=french-quarter-nola`):

```
$ python serving/query_endpoint.py --profile surge-exposure \
    --surge-ft 0.0 --height-m 4.78 --flood-active 0 --region french-quarter-nola
Predicted: $64,510.84
```

That's within 0.1% of the real cell average ($64,575.62) — the trained
model "catches" the exact blind spot the heuristic misses.

**But look at what it had to work with.** Every building in this cell has
`surge_ft = 0.0` and `flood_active = False` — no exceptions, no variance
at all (confirmed via `training_data.csv`). Those two features carry
*zero* information for this region. The only way the model can be this
accurate is by keying almost entirely off the `region_slug_french-quarter-
nola` one-hot and effectively looking up that region's mean payout —
exactly the confound `README.md` already names as a risk with only 13
grid cells ("a model can 'win' mainly by learning per-region average
payouts rather than a real surge/severity relationship").

So the honest framing for wiring this into the agent (next step) isn't
"the model understands NOLA's flood risk the heuristic missed." It's: *the
model has memorized this region's historical average loss, which happens
to be a much better guess than the heuristic's hardcoded zero — but it
would be equally confident and equally right-for-the-wrong-reason for any
other region with a similarly uniform feature profile, and it has learned
nothing that would transfer to a NOLA building the model hasn't
effectively already seen the region-average for.* That query used the
live serving endpoint, which is fit on *all* data including NOLA's own
label — so it can only show what the model memorized, not what it
generalizes to. The systematic check below re-does this properly.

## Systematic check: a general rule, not one anecdote

`map_blind_spots.py` re-runs this across all 13 cells using **honest
out-of-fold predictions** (`GroupKFold` on `cell_id`, identical scheme to
`train_model.py` — a cell held out in a given fold is never seen, its
label included, during that fold's training). This is the fair test of
whether the model would have caught a blind spot *in advance*, not after
memorizing it.

**Headline: the honest, cross-validated model still wins in aggregate.**
Cell-level correlation with real severity: heuristic r = 0.719, model
(out-of-fold) r = **0.843**. That's a real improvement, not leakage.

**But it is not uniformly better, and one case makes that concrete:**

| Cell | Heuristic score | Real severity | Model (honest OOF) | Verdict |
|---|---:|---:|---:|---|
| French Quarter, NOLA | 0.000 | $64,576 | $27,158 | Model closes ~40% of the gap — real improvement, still underrates |
| Galveston Seawall | 0.011 | $52,437 | $29,195 | Same pattern — partial, not full, correction |
| **Clearwater Beach** | 0.128 | $77,257 | **$28,377** | **Model is worse than the heuristic here** — underpredicts by ~63% where the heuristic was already reasonably close |
| Fort Myers Beach (3 cells) | 0.19–0.20 | $85k–$130k | tracks closely | Both already work — this is the county the original study validated on |

Full ranked table: [`blind_spot_map.csv`](blind_spot_map.csv).

**The general rule, then, isn't "prefer the model" or "prefer the
heuristic" — it's "flag disagreement."** Clearwater Beach is the reason a
naive "always trust the bigger/learned number" policy would fail: there,
the simple heuristic beats a legitimately, honestly cross-validated
model. What both the NOLA/Galveston cases and the Clearwater
counter-example have in common is that they're all places where the two
methods *disagree sharply* — which is exactly the signal worth surfacing
to a human (or an agent's write tool) automatically, regardless of which
side turns out to be right. That's the concrete design input for wiring
this into `surge-exposure-agent`: report both numbers, don't silently
prefer either, and treat large disagreement itself as the trigger for
review.

Full per-cell tables: [`8region_grid.csv`](8region_grid.csv),
[`blind_spot_map.csv`](blind_spot_map.csv).
