# PCI LPAPS cutoffs per concept and direction

Reference snapshot (2026-08-29) of the benchmark's preservation cutoffs, kept here in
case the PCI cells themselves are lost. The protocol truncates every method's
alignment–LPAPS curve at **cutoff = min(PCI-all, PCI-loc) max-LPAPS**, per concept and
direction (see `auc.py: global_cutoff` / `pci_cutoff`). For every PCI cell the max is
attained at the fixed grid endpoint |alpha| = 30 (a full prompt swap), so these values
are equivalently "PCI's LPAPS at the complete prompt replacement".

Source cells: `pci_{all,loc}_<concept>/protocol_results/lpaps.csv` under the eval root
of the cluster that benchmarked the concept (WCSS `outputs/eval` for the 5 core /
transferred concepts, Athena `outputs/eval_athena` for the 4 others). The vocal_gender
and vocal_style rescored passes (`protocol_results_fvs` / `_rv`) copy `lpaps.csv`
verbatim, so their cutoffs are identical to these.

**Bold** = min of the pair = the cutoff in force.

| concept | direction | cluster | PCI_all_LPAPS | PCI_loc_LPAPS |
|---|---|---|---|---|
| piano | pos | WCSS | 3.8059 | **3.8039** |
| piano | neg | WCSS | 3.5434 | **3.4916** |
| mood | pos | WCSS | 3.5008 | **3.2392** |
| mood | neg | WCSS | 4.0356 | **3.8762** |
| tempo | pos | WCSS | 3.6498 | **3.6331** |
| tempo | neg | WCSS | 3.9234 | **3.6615** |
| vocal_gender | pos | WCSS | **4.0329** | 4.0954 |
| vocal_gender | neg | WCSS | **3.7686** | 3.8428 |
| vocal_style | pos | WCSS | **4.4401** | 4.5197 |
| vocal_style | neg | WCSS | **3.3296** | 3.4504 |
| violin | pos | Athena | 4.0837 | **4.0731** |
| violin | neg | Athena | 3.5613 | **3.4967** |
| guitar_electronic | pos | Athena | 3.9149 | **3.8596** |
| guitar_electronic | neg | Athena | **3.5209** | 3.5262 |
| rock_genre | pos | Athena | 4.3933 | **4.3331** |
| rock_genre | neg | Athena | 4.2190 | **4.2033** |
| electronic_music | pos | Athena | 4.2371 | **4.2151** |
| electronic_music | neg | Athena | 4.1802 | **4.1712** |

Known consumers of pinned copies of these values: the Athena-concept cutoffs hard-coded
in `sh_scripts/steering/slurm_pwr/calib_sae_benchmark_pwr.sh` (WCSS has no PCI cells
for those 4) and `scripts/sweeps/sae_topk2d{,_neg}_pci/sweep_grid.py: pci_cutoff()`,
which resolves them live from the mounts.

Regenerate this table from the mounts:
`get_lpaps_max(<root>/pci_{all,loc}_<c>/protocol_results, <dir>)` from
`src/steering/eval/auc.py`, over both roots and directions.
