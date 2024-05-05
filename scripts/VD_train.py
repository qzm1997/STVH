import sys
sys.path.append(".")
from train import *
from config import Config
cfg=Config('volleyball')
cfg.inference_module_name = 'STVH_volleyball'
cfg.cuda = "cuda:0"
cfg.device_list="0,1"
cfg.list = [0,1]
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2  
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 2
cfg.num_features_boxes = 1024
# vgg16 setup
cfg.backbone = 'vgg16'
cfg.out_size = 22, 40
cfg.emb_features = 512
cfg.model_path = '/mnt/qzm/DIN-Group-Activity-Recognition-Benchmark-main/result/[Dynamic Volleyball_stage2]<2023-12-07_21-05-02>/stage2_epoch42_95.51%.pth'


# cfg.train_dropout_prob = 0.5
cfg.batch_size = 2
cfg.test_batch_size = 1
cfg.num_frames = 12
cfg.load_model = True
cfg.train_learning_rate = 1e-5
cfg.lr_plan = {15: 5e-6}
cfg.max_epoch = 60
# cfg.max_epoch=120
cfg.train_dropout_prob = 0.5
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
# cfg.activity_weight = [1. ,1. ,1. ,2. ,1. ,1. ,1., 2.]
# cfg.warm_up_iter = 0
# cfg.T_max = 30	# 周期
# cfg.lr_max = 1e-5	# 最大值
# cfg.lr_min = 1e-7	# 最小值

cfg.exp_note = 'STVH Volleyball'
cfg.init_config()
train_net(cfg)


