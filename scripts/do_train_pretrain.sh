# pretrain on image transformations 
data_path=./dataset
config_name=pretrain
run_id=0
train_collection=pretrain_train
val_collection=pretrain_val

python main.py --data_path $data_path --train_collection $train_collection --val_collection $val_collection \
--config_name $config_name --run_id $run_id \
--pretrain