# bolt_nut_detection
Use detectron2 detection the bolts and nuts.
1. 生成 COCO 格式的 JSON文件件
    cd src/data 
    python origin_to_coco.py --help 查看各个参数的说明
    说明：此处支持特定格式转换，具体原始标签格式参见datasets/lslm-test/eval.txt
2. 训练过程参见jupyter notebook
    cd src/model
    train_lslm.ipynb 包含具体训练过程
3. 测试参见jupyter notebook
    cd src/model
    evaluating_prediction_model.ipynb 包含具体测试过程
    
