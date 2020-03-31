"""
# 复制扩充test datasets
"""

import os
import cv2
import shutil
import copy

class DatasetCopyExpand(object):
    def __init__(self, origin_label_file='../../datasets/lslm-test/eval.txt',
                        origin_images_dir = '../../datasets/lslm-test',
                        output_label_file='../../datasets/lslm-test/eval_expand.txt',
                        expand_turn=4):
        self.origin_label_file = origin_label_file
        self.origin_images_dir = origin_images_dir
        self.output_label_file = output_label_file
        self.expand_turn = expand_turn

    def image_copy_expand(self):
        with open(self.origin_label_file, mode='r', encoding='utf-8') as fp:
            lines = fp.readlines()
        with open(self.output_label_file, mode='w', encoding='utf-8') as fw:
            for line in lines:
                fw.write(line)

            # 原始图片数
            origin_images = len(lines)
            # 复制扩充expand_turn轮
            for i in range(1, self.expand_turn+1):
                for index, line in enumerate(lines):
                    new_image_name = "{}.jpg".format(str(origin_images*i + index+1))
                    print(new_image_name)
                    new_image_path = os.path.join(self.origin_images_dir, new_image_name)
                    origin_image_name = line.split('\t')[0].strip('')
                    print(origin_image_name)
                    origin_image_path = os.path.join(self.origin_images_dir, origin_image_name)
                    # 复制图片
                    shutil.copy(origin_image_path, new_image_path)
                    # 将new image label写入文件
                    new_line = copy.deepcopy(line)
                    new_line = new_line.replace(origin_image_name, new_image_name)
                    fw.write(new_line)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-OLF", "--originLableFile", dest="originLableFile",
                        default='../../datasets/lslm-test/eval.txt',
                        type=str, help="原始图片标签文件路径.")
    parser.add_argument("-IP", "--imagePath", dest="imagePath",
                        default="../../datasets/lslm-test",
                        type=str, help="图片路径.")
    parser.add_argument("-OPLF", "--outputLabelFile", dest="outputLabelFile",
                        default="../../datasets/lslm-test/eval_expand.txt",
                        type=str, help="扩张之后图片标签文件保存路径.")
    args = parser.parse_args()

    pt = DatasetCopyExpand(args.originLableFile, args.imagePath, args.outputLabelFile)
    pt.image_copy_expand()
