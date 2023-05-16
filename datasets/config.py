# ------------------------------------------------------------------------------
# Based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------


from yacs.config import CfgNode as CN

config = CN()
#
# config.OUTPUT_DIR = ''
# config.LOG_DIR = ''
config.GPUS = (0,)
config.WORKERS = 1
# config.PRINT_FREQ = 20
# config.AUTO_RESUME = False
# config.PIN_MEMORY = True
# config.RANK = 0
#
# # Cudnn related params
# config.CUDNN = CN()
# config.CUDNN.BENCHMARK = True
# config.CUDNN.DETERMINISTIC = False
# config.CUDNN.ENABLED = True
#
# # common params for NETWORK
config.MODEL = CN()
# config.MODEL.NAME = 'seg_hrnet'
# config.MODEL.PRETRAINED = ''
config.MODEL.ALIGN_CORNERS = True
config.MODEL.NUM_OUTPUTS = 1
# config.MODEL.EXTRA = CN(new_allowed=True)
#
# config.MODEL.OCR = CN()
# config.MODEL.OCR.MID_CHANNELS = 512
# config.MODEL.OCR.KEY_CHANNELS = 256
# config.MODEL.OCR.DROPOUT = 0.05
# config.MODEL.OCR.SCALE = 1
#
# config.LOSS = CN()
# config.LOSS.USE_OHEM = False
# config.LOSS.OHEMTHRES = 0.9
# config.LOSS.OHEMKEEP = 100000
# config.LOSS.CLASS_BALANCE = False
# config.LOSS.BALANCE_WEIGHTS = [1]
#
# # DATASET related params
config.DATASET = CN()
config.DATASET.ROOT = '/home/htessier/PycharmProjects/semantic_segmentation/cityscapes/'
config.DATASET.DATASET = 'cityscapes'
config.DATASET.NUM_CLASSES = 19
config.DATASET.TRAIN_SET = './list/cityscapes/train.lst'
# config.DATASET.EXTRA_TRAIN_SET = ''
config.DATASET.TEST_SET = './list/cityscapes/val.lst'
#
# # training
config.TRAIN = CN()
#
# config.TRAIN.FREEZE_LAYERS = ''
# config.TRAIN.FREEZE_EPOCHS = -1
# config.TRAIN.NONBACKBONE_KEYWORDS = []
# config.TRAIN.NONBACKBONE_MULT = 10
#
config.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
config.TRAIN.BASE_SIZE = 2048
config.TRAIN.DOWNSAMPLERATE = 1
config.TRAIN.FLIP = True
config.TRAIN.MULTI_SCALE = True
config.TRAIN.SCALE_FACTOR = 16
#
config.TRAIN.RANDOM_BRIGHTNESS = False
config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10
#
# config.TRAIN.LR_FACTOR = 0.1
# config.TRAIN.LR_STEP = [90, 110]
# config.TRAIN.LR = 0.01
# config.TRAIN.EXTRA_LR = 0.001
#
# config.TRAIN.OPTIMIZER = 'sgd'
# config.TRAIN.MOMENTUM = 0.9
# config.TRAIN.WD = 0.0001
# config.TRAIN.NESTEROV = False
config.TRAIN.IGNORE_LABEL = -1
#
# config.TRAIN.BEGIN_EPOCH = 0
# config.TRAIN.END_EPOCH = 484
# config.TRAIN.EXTRA_EPOCH = 0
#
# config.TRAIN.RESUME = False
#
config.TRAIN.BATCH_SIZE_PER_GPU = 1
config.TRAIN.SHUFFLE = True
# # only using some training samples
# config.TRAIN.NUM_SAMPLES = 0
#
# # testing
config.TEST = CN()
#
config.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
config.TEST.BASE_SIZE = 2048
#
# config.TEST.BATCH_SIZE_PER_GPU = 32
# # only testing some samples
config.TEST.NUM_SAMPLES = 0
#
# config.TEST.MODEL_FILE = ''
# config.TEST.FLIP_TEST = False
# config.TEST.MULTI_SCALE = False
# config.TEST.SCALE_LIST = [1]
#
config.TEST.OUTPUT_INDEX = -1
#
# # debug
# config.DEBUG = CN()
# config.DEBUG.DEBUG = False
# config.DEBUG.SAVE_BATCH_IMAGES_GT = False
# config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
# config.DEBUG.SAVE_HEATMAPS_GT = False
# config.DEBUG.SAVE_HEATMAPS_PRED = False
#
#
# def update_config(cfg, args):
#     cfg.defrost()
#
#     cfg.merge_from_file(args.cfg)
#     cfg.merge_from_list(args.opts)
#
#     cfg.freeze()
#
#
# if __name__ == '__main__':
#     import sys
#
#     with open(sys.argv[1], 'w') as f:
#         print(config, file=f)
