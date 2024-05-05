import sys
sys.path.append(".")
from train import *
import os
from config import Config
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cfg=Config('collective')
cfg.inference_module_name = 'STVH_collective'
cfg.cuda = "cuda:0"
cfg.device_list="0,1"
cfg.list = [0,1]
cfg.training_stage=2
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.train_backbone = True
cfg.load_model = False
cfg.num_features_boxes = 1024
cfg.backbone = 'res18'
cfg.image_size = 480, 720
cfg.out_size = 15, 23
cfg.emb_features = 512
cfg.model_path = ''
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.model_path = '/mnt/qzm/111/DIN-Group-Activity-Recognition-Benchmark-main/result/[Dynamic_collective_stage2]<2023-12-27_18-35-58>/stage2_epoch11_98.29%.pth'
cfg.batch_size = 2
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 1e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 30
cfg.r1 = 0.1
cfg.r2 = 0.5
cfg.test_before_train=False
cfg.exp_note='STVH_collective'
cfg.init_config()
train_net(cfg)