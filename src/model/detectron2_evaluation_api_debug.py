"""
# 使用detectron2 API 进行模型评测
"""
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.config import get_cfg

# 导入基本detectorn2 API函数进行模型评测
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.evaluation import inference_on_dataset

# 注册数据集
register_coco_instances('lslm_coco_test', {},
                        '/tmp/pycharm_project_78/datasets/annotations/lslm_test.json',
                       '/tmp/pycharm_project_78/datasets/lslm-test')

def setup_cfg(args):
    """设置基本参数"""
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.DATASETS.TRAIN = ("lslm_coco_train", )
    cfg.DATASETS.TEST = ("lslm_coco_test",)
    # cfg.DATASETS.TEST = ("lslm_coco_train", )
    cfg.DATALOADER.NUM_WORKERS = 2

    # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = args.model_weights

    cfg.SOLVER.IMS_PER_BATCH = 2
    # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00025
    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 300
    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # coco datasets(bolt, nut)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RPN.NMS_THRESH = 0.5

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def lslm_evaluation(cfg, output_dir=None):
    # 创建模型
    model = build_model(cfg)
    # 创建数据器
    print(cfg.DATASETS.TEST[0])
    evaluator_type = MetadataCatalog.get(cfg.DATASETS.TEST[0]).evaluator_type
    lslm_evaluation_data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    # for idx, inputs in enumerate(lslm_evaluation_data_loader):
    #     model.eval()
    #     outputs = model(inputs)
    # print("outputs: ", outputs)
    # 创建评估器
    if output_dir is None:
        print(cfg.OUTPUT_DIR)
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    lslm_coco_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, True, output_dir=output_dir)
    # 进行数据评估
    evaluation_result = inference_on_dataset(model, lslm_evaluation_data_loader, lslm_coco_evaluator)
    # 下面这种做法并没有什么用
    # evaluation_result = inference_on_dataset(model, lslm_evaluation_data_loader,
    #                                          DatasetEvaluators([lslm_coco_evaluator]))

    return evaluation_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSLM test for builtin models")
    parser.add_argument(
        "--config-file",
        default="/tmp/pycharm_project_78/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model-weights",
        default="/tmp/pycharm_project_78/models/model_final.pth",
        type=str,
        help="Initial weights path.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args(args=[])
    print(args.model_weights)
    cfg = setup_cfg(args)

    # 进行模型评估
    evl_res = lslm_evaluation(cfg)

