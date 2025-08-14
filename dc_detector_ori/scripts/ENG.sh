
export CUDA_VISIBLE_DEVICES=1

for i in {1..1};
do

python main.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 16  --mode train --dataset ENG  --data_path ENG   --input_c 2 --output 1 --index $i --win_size 15 --patch_size 35
python main.py --anormly_ratio 0.85 --num_epochs 10   --batch_size 16    --mode test    --dataset ENG   --data_path ENG     --input_c 2   --output 1  --index $i --win_size 15 --patch_size 35
done  

