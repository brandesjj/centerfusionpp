export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src

## Perform detection and evaluation
python test.py ddd \
    --load_model ../models/centerfusionpp.pth \
    --exp_id centerfusionpp \
    --run_dataset_eval \
    --val_split val \
    --gpus 0 \
    --use_sec_heads \
    --use_lfa \
    --use_early_fusion \
    --pointcloud \
    --flip_test \

cd ..

