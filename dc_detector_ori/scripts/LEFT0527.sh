export CUDA_VISIBLE_DEVICES=1

for i in {1..1};
do

python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 128  --mode train --dataset LEFT0527  --data_path LEFT0527   --input_c 8 --output 8 --index $i --win_size 105 --patch_size 357
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 128    --mode test    --dataset LEFT0527   --data_path LEFT0527     --input_c 8   --output 8  --index $i --win_size 105 --patch_size 357

done  

