# init...
#nohup python3 train_ssd.py --dataset_type ava --datasets /home/pi/ava_dataset/ --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5 &
nohup python3 train_ssd.py --num_workers 8 --dataset_type ava --datasets /home/pi/ava_dataset/ --net mb2-ssd-lite --resume models/mb2-ssd-lite-Epoch-0-Loss-4.680627800611676.pth --scheduler cosine --lr 0.001 --t_max 100 --validation_epochs 1 --num_epochs 100 --base_net_lr 0.001  --batch_size 8 &

