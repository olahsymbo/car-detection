import numpy as np
import cv2
import os 
import torch, torchvision
assert torch.__version__.startswith("1.8")
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from google.colab.patches import cv2_imshow
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model, build_roi_heads
from detectron2.data.datasets import load_coco_json

dataNameTr = "carV_train"
dataNameTe = "carV_test"
jsonTrFolder = "/content/drive/MyDrive/car_detection/cocosplit-master/carMake_train.json"
jsonTeFolder = "/content/drive/MyDrive/car_detection/cocosplit-master/carMake_test.json"
imageFolder = "/content/drive/MyDrive/car_detection/carMake/dump"
configFile = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
checkPointurl = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

register_coco_instances(dataNameTr, {}, jsonTrFolder, imageFolder)
register_coco_instances(dataNameTe, {}, jsonTeFolder, imageFolder)
metadata = MetadataCatalog.get(dataNameTr)




class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(configFile))
cfg.DATASETS.TRAIN = (dataNameTr,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkPointurl)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 500 #adjust up if val mAP is still rising, adjust down if overfit
#cfg.SOLVER.STEPS = (1000, 1500)
#cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 17
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "LP.pth")
cfg.DATASETS.TEST = (dataNameTe, )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get(dataNameTe)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "LP.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator(dataNameTe, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, dataNameTe)
inference_on_dataset(trainer.model, val_loader, evaluator)

