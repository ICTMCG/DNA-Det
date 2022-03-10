
# for the celeba experiment
data_path=./dataset
config_name=celeba
run_id=1
device=cuda:0
train_collection=celeba_train
val_collection=celeba_val
pretrain_model_path=./dataset/pretrain_train/models/pretrain_val/pretrain/run_0/model.pth

python3 main.py  --data_path $data_path --train_collection $train_collection --val_collection $val_collection \
--config_name $config_name --run_id $run_id  --device $device \
--pretrain_model_path $pretrain_model_path \
--test_data_path ./dataset/celeba_test/annotations/celeba_test_cross_dataset.txt


# for the lsun-bedroom experiment
data_path=./dataset
config_name=lsun
run_id=1
device=cuda:0
train_collection=lsun_train
val_collection=lsun_val
pretrain_model_path=./dataset/pretrain_train/models/pretrain_val/pretrain/run_0/model.pth

python3 main.py  --data_path $data_path --train_collection $train_collection --val_collection $val_collection \
--config_name $config_name --run_id $run_id  --device $device \
--pretrain_model_path $pretrain_model_path \
--test_data_path ./dataset/lsun_test/annotations/lsun_test_cross_dataset.txt


# for the in the wild experiment
data_path=./dataset
config_name=in_the_wild
run_id=1
device=cuda:0
train_collection=in_the_wild_train
val_collection=in_the_wild_val
pretrain_model_path=./dataset/pretrain_train/models/pretrain_val/pretrain/run_0/model.pth

python3 main.py  --data_path $data_path --train_collection $train_collection --val_collection $val_collection \
--config_name $config_name --run_id $run_id  --device $device \
--pretrain_model_path $pretrain_model_path \
--test_data_path ./dataset/in_the_wild_test/annotations/in_the_wild_test_cross_dataset.txt
