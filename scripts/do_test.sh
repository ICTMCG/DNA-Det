# # for the celeba experiment
# config_name=celeba
# model_path=./dataset/celeba_train/models/celeba_val/celeba/run_0/model.pth
# python3 pred_eval.py --model_path $model_path --config_name $config_name \
# --test_data_paths ./dataset/celeba_test/annotations/celeba_test.txt \
# ./dataset/celeba_test/annotations/celeba_test_cross_seed.txt \
# ./dataset/celeba_test/annotations/celeba_test_cross_loss.txt \
# ./dataset/celeba_test/annotations/celeba_test_cross_finetune.txt \
# ./dataset/celeba_test/annotations/celeba_test_cross_dataset.txt  

# # for the lsun-bedroom experiment
# config_name=lsun
# model_path=./dataset/lsun_train/models/lsun_val/lsun/run_0/model.pth
# python3 pred_eval.py --model_path $model_path --config_name $config_name \
# --test_data_paths ./dataset/lsun_test/annotations/lsun_test.txt \
# ./dataset/lsun_test/annotations/lsun_test_cross_seed.txt \
# ./dataset/lsun_test/annotations/lsun_test_cross_loss.txt \
# ./dataset/lsun_test/annotations/lsun_test_cross_finetune.txt \
# ./dataset/lsun_test/annotations/lsun_test_cross_dataset.txt 

# for the in the wild experiment
config_name=in_the_wild
model_path=./dataset/in_the_wild_train/models/in_the_wild_val/in_the_wild/run_0/model.pth
python3 pred_eval.py --model_path $model_path --config_name $config_name \
--test_data_paths ./dataset/in_the_wild_test/annotations/in_the_wild_test.txt \
./dataset/in_the_wild_test/annotations/in_the_wild_test_cross_dataset.txt


