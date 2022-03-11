
config_name=celeba
model_path=./dataset/celeba_train/models/celeba_val/celeba/run_0/model.pth

python3 draw_tsne.py --model_path $model_path --config_name $config_name \
--draw_data_path ./dataset/celeba_test/annotations/allcross_test.txt \
--do_extract \
--do_fit \
--save_name cross_all \
--legend real ProGAN_cross_seed MMDGAN_cross_ft SNGAN_cross_data InfoMaxGAN_cross_loss