# UncertaintyTrack
Pipeline for MOT model evaluation built on UncertaintyTrack repository

## Training:
example
```
python3.10 train.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --work-dir ./work_dirs/test_run --no-validate \
    --cfg-options data.workers_per_gpu=1 \
    model.detector.init_cfg.checkpoint=checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
```

## Test / evaluation:
### Eval mode: outputs evaluation metrics
example
```
nohup python3.10 test.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --eval track \
    --out results.pkl > nohup_test_added_test_dataset_in_config.out
```

### Formate only mode: outputs annotation files from inference as well as annotated videos
example
```
nohup python3.10 test.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --out results.pkl \
    --format-only > nohup_test_full_test_dataset.out
```

## Custom config File with fixed covariance value
path: configs/custom/pseudo_uncertainmot_yolox_x_3x6_mot17-half.py
command to run test.py:
```
nohup python3.10 test.py configs/custom/pseudo_uncertainmot_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --out results_custom.pkl \
    --format-only > nohup_custom.out
```