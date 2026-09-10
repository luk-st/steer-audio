# Published results

The paper's numbers, frozen so the tables and curves can be rebuilt without
re-running any generation or scoring.

| file | rows | contents |
|---|---|---|
| `steering_metrics.csv` | 20,925 | per-alpha `lpaps`, `muqt`, `clap` for every method × concept × preservation axis |
| `quality_metrics.csv` | 4,185 | per-alpha Audiobox aesthetics (CE / CU / PC / PQ) |
| `localization_impact.csv` | 1,296 | layer impact `I(l, c)` per model, concept and attention block |
| `tables_lpaps.md`, `tables_localization.md` | — | reference output, to check a rebuild reproduces |

`cell` names the arm (`caa_all`, `caa_loc`, `sae`, …) and `axis` the
preservation metric: `lpaps` is the one used in the main tables, the other four
(`harmony`, `melody`, `rhythm`, `ssm`) are the decomposed axes. Direction is the
sign of `alpha`. Values are means over the evaluation prompts, rounded to 6
decimals; the per-prompt arrays and standard deviations are not shipped.

## Rebuilding

```bash
python scripts/results/make_tables.py --out tables.md   # per-concept + average
python scripts/results/make_tables.py --axis harmony    # a decomposed axis
python scripts/results/make_tables.py --localization    # layer impact
python scripts/results/plot_curves.py                   # pres. vs Δalignment
```

Both scripts call `expand.py`, which materialises these CSVs into the
`protocol_results/` layout in a temp dir, so the AUC protocol is applied by
`src/steering/eval/auc.py` itself — the same code that produced the paper —
rather than a second implementation. Pass `--root <eval-root>` to point them at
a live sweep instead.
