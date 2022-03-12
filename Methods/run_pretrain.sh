gpu=0

# twitter pretrain
lr=0.001
wd=1e-5
CUDA_VISIBLE_DEVICES=0 python -u PREP_TCN_pretrain.py --cuda --cuda_index $gpu --max_try 50 --lr $lr --file_dir "../data/twitter/" --filename_train "pretrain_data_train.txt" --filename_val "pretrain_data_val.txt" --interval_time 5 --ksize 8 --mlp_hid_size 8 --levels 12 --dropout 0.0 --weight_decay $wd --epochs 100 --clip -1 --nhid 8 --max_label 18 --time_slices 1800 --dataset "twitter"
