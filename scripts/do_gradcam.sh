
config_name=celeba
model_path=./dataset/celeba_train/models/celeba_val/celeba/run_0/model.pth

python3 draw_gradcam.py --model_path $model_path --config_name $config_name \
--draw_data_path ./dataset/celeba_test/annotations/celeba_test.txt \
--layer_name main.3.main.2