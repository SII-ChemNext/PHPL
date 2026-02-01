# PEMAL Workflow

This repository contains the preprocessing, pre-training, and fine-tuning pipeline used for the PEMAL experiments. The core training code lives in the `chem ` directory (note the trailing space in the folder name); use quotes when referencing it from the shell.

## Environment
- Python 3.10+
- PyTorch with GPU support
- torch_geometric
- RDKit (required by `desc.py`)
- tqdm, pandas, numpy, scikit-learn, wandb

Install the Python dependencies in your preferred environment before running the steps below.

## End-to-end run
1. **Add descriptors**  
   Configure the input/output CSV paths at the top of `desc.py`, then run from the repository root:  
   ```bash
   python desc.py
   ```  
   This may take a while; it writes descriptor-enriched CSVs to the paths you configured.

2. **Pre-train**  
   Launch the multi-dataset pre-training sweep (takes the longest):  
   ```bash
   bash "chem /pretrain.sh"
   ```  
   Results are written under `results/20250913/pretrain_beta3/` by default. Adjust the hyper-parameter grids or output path in the script as needed.

3. **Fine-tune**  
   Point `pretrain_checkpoint` in `chem /finetune.sh` to the `best_model.pth` produced in the pre-training step, then run:  
   ```bash
   bash "chem /finetune.sh"
   ```  
   Fine-tuned checkpoints are saved to `results/20251023/final_pre_auc_noisy/` by default.

## Notes
- Run the shell scripts from the repository root so the relative paths resolve correctly. Because the directory name includes a trailing space (`chem `), remember to wrap it in quotes when calling the scripts.
- The provided paths and hyper-parameter grids mirror the current experiments; feel free to edit them to suit new runs.
