export CUDA_VISIBLE_DEVICES=1

for i in {1..16};
do

python main.py --anormly_ratio 5 --num_epochs 3   --batch_size 128  --mode train --dataset FLY  --data_path FLY  --input_c 19 --output 19 --index $i --win_size 105 --patch_size 357
python main.py --anormly_ratio 5 --num_epochs 10   --batch_size 128    --mode test    --dataset FLY   --data_path FLY  --input_c 19   --output 19  --index $i --win_size 105 --patch_size 357

done  

