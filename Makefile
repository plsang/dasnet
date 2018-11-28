DATASETS=cityscapes bdd
DATASET?=cityscapes

ifeq ($(DATASET), cityscapes)
	DATA_DIR=/datashare/datasets/cityscapes
	CROP_SIZE=769
else ifeq ($(DATASET), bdd)
	DATA_DIR=/datashare/datasets/bdd/bdd100k
	CROP_SIZE=512
else
    @echo 'Unknown $(DATASET)!!!'
endif

# OUT_DIR=output
OUT_DIR=/users/sang/works/dasnet/output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model
LOG_DIR=$(OUT_DIR)/logs


SPLITS=train val test

MODEL_FILE=$(OUT_DIR)/model/model_lr0001.pth
RESULT_FILE=$(OUT_DIR)/result/output.json

PRETRAIN_MODEL_DIR=/users/sang/models
START_FROM=$(PRETRAIN_MODEL_DIR)/resnet101-imagenet.pth

MODEL_TYPES=baseline base_oc_dsn pyramid_oc_dsn asp_oc_dsn

metadata: $(patsubst %, $(META_DIR)/$(DATASET)_%.txt,$(SPLITS))
$(META_DIR)/$(DATASET)_%.txt:
	python src/metadata.py \
		--output_file $@ \
		--data_root $(DATA_DIR) \
		--split $* \
		--dataset $(DATASET)

train: $(patsubst %,$(OUT_DIR)/model/$(DATASET)_%_lr0001.pth,$(MODEL_TYPES))
$(OUT_DIR)/model/$(DATASET)_%_lr0001.pth:
	python src/train.py \
		--output_file $@ \
		--data_dir '' \
		--train_data_list $(META_DIR)/$(DATASET)_train.txt \
		--val_data_list $(META_DIR)/$(DATASET)_val.txt \
		--start_from $(START_FROM) \
		--random_mirror --random_scale \
		--crop_size_h $(CROP_SIZE) --crop_size_w $(CROP_SIZE) \
		--num_workers 8 --batch_size 16 \
		--num_epochs 100 --max_patience 100 --learning_rate 0.0001 --lr_update 100 \
		--cnn_type resnet101 --model_type $* --dataset $(DATASET) \
		2>&1 | tee $(LOG_DIR)/train_$(DATASET)_$*_lr0001.log 

test: $(RESULT_FILE)
$(RESULT_FILE): $(MODEL_FILE)
	python src/test.py $^ \
		--test_data_list $(OCNET_ROOT)/dataset/list/cityscapes/val.lst \
		--crop_size_h 1024 --crop_size_w 2048 \
		--batch_size 1 --num_workers 8\
		--output_file $@ 


