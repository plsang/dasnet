# IN_DIR=/nas/datashare/datasets/tusimple-benchmark

IN_DIR=/datashare/datasets/cityscapes 
# OUT_DIR=output
OUT_DIR=/users/sang/works/drivablenet/output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model

#MODEL_FILE=$(OUT_DIR)/model/model_b16.pth
#MODEL_FILE=$(OUT_DIR)/model/model_lr001.pth
MODEL_FILE=$(OUT_DIR)/model/model_lr0001.pth
RESULT_FILE=$(OUT_DIR)/result/output.json

OCNET_ROOT=/users/sang/clones/OCNet

START_FROM=$(OCNET_ROOT)/pretrained_model/resnet101-imagenet.pth

metadata:
	echo 'Not implemented yet'

train: $(MODEL_FILE)
$(MODEL_FILE): 
	python src/train.py \
		--output_file $@ \
		--data_dir $(IN_DIR) \
		--train_data_list $(OCNET_ROOT)/dataset/list/cityscapes/train.lst \
		--val_data_list $(OCNET_ROOT)/dataset/list/cityscapes/val.lst \
		--start_from $(START_FROM) \
		--random_mirror --random_scale \
		--crop_size_h 769 --crop_size_w 769 \
		--num_workers 8 --batch_size 16 \
		--num_epochs 100 --max_patience 50 --learning_rate 0.0001 --lr_update 50 \
		--cnn_type resnet101 

test: $(RESULT_FILE)
$(RESULT_FILE): $(MODEL_FILE)
	python src/test.py $^ \
		--test_data_list $(OCNET_ROOT)/dataset/list/cityscapes/val.lst \
		--crop_size_h 1024 --crop_size_w 2048 \
		--batch_size 1 --num_workers 8\
		--output_file $@ 

test_1: $(RESULT_FILE)
$(RESULT_FILE): $(MODEL_FILE)
	python src/test.py $^ \
		--data_dir /users/yizhou/rosbags/top_down_view_university_back_0 \
		--crop_size_h 1024 --crop_size_w 2048 \
		--batch_size 1 --num_workers 8\
		--output_file $@ 
