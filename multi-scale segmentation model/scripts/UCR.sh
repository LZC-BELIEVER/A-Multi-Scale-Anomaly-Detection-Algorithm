export CUDA_VISIBLE_DEVICES=1

for i in {1..1};
do

#python main.py --anormly_ratio 5 --num_epochs 3   --batch_size 16  --mode train --dataset UCR  --data_path UCR   --input_c 1 --output 1 --index $i --win_size 90 --patch_size 56
python main.py --anormly_ratio 5 --num_epochs 10   --batch_size 32    --mode test    --dataset UCR   --data_path UCR     --input_c 1   --output 1  --index $i --win_size 90 --patch_size 56

done  

