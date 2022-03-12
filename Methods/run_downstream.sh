gpu=0
wd=1e-5

#twitter ob 1hour,pre 1days
backbone_lr=5e-5
lr=5e-5
CUDA_VISIBLE_DEVICES=0 python -u PREP_TCN_downstream.py --cuda --cuda_index $gpu --max_try 10 --lr $lr --backbone_lr $backbone_lr --file_dir "../data/twitter/" --filename_train "downstream_data_train.txt" --filename_val "downstream_data_val.txt" --filename_test "downstream_data_test.txt" --interval_time 5  --ob_time 3600 --pre_time 86400 --reg 1 --ksize 8 --levels 12 --dropout 0.0 --weight_decay $wd --epochs 1000 --clip -1 --nhid 8 --filename_pretrain_model "./save_models/pretrain_twitter_model_max_label18_time_slices1800_lr0.001_wd5e-05.pkl" --max_label 18 --time_slices 1800 --mlp_hid_size 8 --if_fix 0

#twitter ob 1hour,pre 7days
backbone_lr=5e-5
lr=0.0005
CUDA_VISIBLE_DEVICES=0 python -u PREP_TCN_downstream.py --cuda --cuda_index $gpu --max_try 10 --lr $lr --backbone_lr $backbone_lr --file_dir "../data/twitter/" --filename_train "downstream_data_train.txt" --filename_val "downstream_data_val.txt" --filename_test "downstream_data_test.txt" --interval_time 5  --ob_time 3600 --pre_time 604800 --reg 1 --ksize 8 --levels 12 --dropout 0.0 --weight_decay $wd --epochs 1000 --clip -1 --nhid 8 --filename_pretrain_model "./save_models/pretrain_twitter_model_max_label18_time_slices1800_lr0.001_wd5e-05.pkl" --max_label 18 --time_slices 1800 --mlp_hid_size 8 --if_fix 0

#twitter ob 2hour,pre 7days
backbone_lr=0.0001
lr=0.0001
CUDA_VISIBLE_DEVICES=0 python -u PREP_TCN_downstream.py --cuda --cuda_index $gpu --max_try 10 --lr $lr --backbone_lr $backbone_lr --file_dir "../data/twitter/" --filename_train "downstream_data_train.txt" --filename_val "downstream_data_val.txt" --filename_test "downstream_data_test.txt" --interval_time 5  --ob_time 7200 --pre_time 604800 --reg 1 --ksize 8 --levels 12 --dropout 0.0 --weight_decay $wd --epochs 1000 --clip -1 --nhid 8 --filename_pretrain_model "./save_models/pretrain_twitter_model_max_label18_time_slices1800_lr0.001_wd5e-05.pkl" --max_label 18 --time_slices 1800 --mlp_hid_size 8 --if_fix 0

