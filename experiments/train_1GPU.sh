 #!/bin/bash         

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src
# train
python main.py ddd \
    --load_model ../models/earlyfusion.pth \
    --exp_id centerfusionpp \
    --run_dataset_eval \
    --val_split val \
    --val_intervals 10 \
    --train_split train \
    --shuffle_train \
    --batch_size 16 \
    --num_epochs 50 \
    --lr 1.25e-4 \
    --lr_step 100 \
    --save_point 20,40 \
    --gpus 0 \
    --use_sec_heads \
    --use_lfa \
    --use_early_fusion \
    --pointcloud \
    --freeze_layers \

cd ..
