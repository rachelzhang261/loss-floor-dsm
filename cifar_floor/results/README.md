# Saved-result map

These JSON files are copied unchanged from the verified working project.

| File | Role in the current paper |
|---|---|
| `wide240k_paired_baselines.json` | Table 1 and the reference curves in Figure 1. `tail_integrals` contains the risks and paired improvements on log-SNR [13,13.8]. |
| `wide240k_audit_seed1_grid219.json` | Independent Jacobian/Stein audit for the 240k checkpoint and the Jacobian curve in Figure 1. |
| `wide_extended_audit_grid219.json` | Paired diagnostics across the extended-run checkpoints. |
| `wide240k_dsm_seed_check.json` | Additional held-out DSM sampling check. |
| `wide_extended_grid219.json` | Earlier evaluation of the 60k/120k/240k extended run; supporting record, not an oracle floor measurement. |
| `wide_extended_grid219_summary.json` | Summary of the preceding file, with its denoiser-bias caveat retained. |
| `imagenet_edm_primary.json` | Section 5.3: 512 validation images, 404 classes; risks and paired reductions in nats per scalar dimension. |

`make_audit_figure.py` reads the first two files. Their dimensions and units differ
from the ImageNet record, which reports per-scalar-dimension risk.

Historical schema names are not scientific claims: `floor` and `pooled_floor`
mean learned plug-in estimates in network evaluations; `true` means the common
narrow-range DSM score. The revised manuscript rejects treating those learned
estimates as automatically equal to the Bayes floor. Superseded ranking plots,
initial `analysis*.json` files, early grid reanalyses, and pilot ImageNet outputs
are omitted from this final code package.
