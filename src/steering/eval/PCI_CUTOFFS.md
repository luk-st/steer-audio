# PCI LPAPS cutoffs per concept and direction

Reference snapshot (2026-08-29) of the benchmark's preservation cutoffs, kept here in
case the PCI cells themselves are lost. The protocol truncates every method's
alignment–LPAPS curve at **cutoff = min(PCI-all, PCI-loc) max-LPAPS**, per concept and
direction (see `auc.py: global_cutoff` / `pci_cutoff`). For every PCI cell the max is
attained at the fixed grid endpoint |alpha| = 30 (a full prompt swap), so these values
are equivalently "PCI's LPAPS at the complete prompt replacement".

Source cells: `pci_{all,loc}_<concept>/protocol_results/lpaps.csv` under the eval
root. The vocal_gender and vocal_style rescored passes (`protocol_results_fvs` /
`_rv`) copy `lpaps.csv` verbatim, so their cutoffs are identical to these.

**Bold** = min of the pair = the cutoff in force.

| concept | direction | PCI_all_LPAPS | PCI_loc_LPAPS |
|---|---|---|---|
| piano | pos | 3.8059 | **3.8039** |
| piano | neg | 3.5434 | **3.4916** |
| mood | pos | 3.5008 | **3.2392** |
| mood | neg | 4.0356 | **3.8762** |
| tempo | pos | 3.6498 | **3.6331** |
| tempo | neg | 3.9234 | **3.6615** |
| vocal_gender | pos | **4.0329** | 4.0954 |
| vocal_gender | neg | **3.7686** | 3.8428 |
| vocal_style | pos | **4.4401** | 4.5197 |
| vocal_style | neg | **3.3296** | 3.4504 |
| violin | pos | 4.0837 | **4.0731** |
| violin | neg | 3.5613 | **3.4967** |
| guitar_electronic | pos | 3.9149 | **3.8596** |
| guitar_electronic | neg | **3.5209** | 3.5262 |
| rock_genre | pos | 4.3933 | **4.3331** |
| rock_genre | neg | 4.2190 | **4.2033** |
| electronic_music | pos | 4.2371 | **4.2151** |
| electronic_music | neg | 4.1802 | **4.1712** |

Regenerate this table with `get_lpaps_max(<root>/pci_{all,loc}_<c>/protocol_results,
<direction>)` from `src/steering/eval/auc.py`.
