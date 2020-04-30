nohup python3 -u eval_ssd.py --dataset_type ava --dataset /home/pi/ava_dataset/ --net mb2-ssd-lite --trained_model models/mb2-ssd-lite-Epoch-0-Loss-4.680627800611676.pth --label_file models/ava-model-labels.txt  > eval_result.log &

