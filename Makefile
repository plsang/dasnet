# IN_DIR=/nas/datashare/datasets/tusimple-benchmark

IN_DIR=/datashare/datasets/cityscapes 
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model

MODEL_FILE=$(OUT_DIR)/model/model.pth

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
		--start-from $(START_FROM) \
		--random-mirror --random-scale \
		--num_workers 8 --batch_size 16 --num_epochs 50 \
		--cnn_type resnet101 

