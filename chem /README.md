# PEMAL Chem Scripts

Key training utilities for PEMAL live here. See the repository root `README.md` for the full workflow and environment setup.

- `pretrain.sh` runs the descriptor-based pre-training loop (`pretrain_desc.py`).
- `finetune.sh` runs the downstream fine-tuning loop (`finetune.py`).

Run the scripts from the repository root, for example:

```bash
bash "chem /pretrain.sh"
bash "chem /finetune.sh"
```
