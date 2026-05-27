# Examples

Validate the tiny bundled sample:

```bash
python tools/validate_sample_dataset.py \
  --anno_path examples/sample_data/sample_mo3d.json \
  --data_path examples/sample_data/sample_point_clouds \
  --pointnum 8
```

Check that train and eval commands expand without cluster-specific paths:

```bash
DRY_RUN=1 scripts/train/train_joint.sh
DRY_RUN=1 scripts/eval/infer.sh
```
