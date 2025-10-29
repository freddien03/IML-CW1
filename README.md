## Setup
Run `pip install numpy matplotlib`

## Data
If not present, place data files at:
- `wifi_db/clean_dataset.txt`
- `wifi_db/noisy_dataset.txt`

## Run
Run `python3 main.py`
- Prints 10-fold metrics for clead and noisy data
- Saves confusion matrices in simple text form at `clean_confusion_matrix.txt`, `noisy_confusion_matrix.txt`
- Saves tree image for the clean model at `clean_tree.png`

## Notes
- Change `RNG_SEED` in `main.py` if needed to adjust randomization
- Visual layout tweaks in `save_tree_png(...)`, adjust `hsep`, `vstep`, `figsize`, and font/padding if desired