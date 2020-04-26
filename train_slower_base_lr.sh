nohup python3 train_ssd.py --dataset_type ava --datasets /home/pi/ava_dataset/ --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5 &

