"""
# 原始存储数据的TXT文件转为coco格式的json文件
"""
import os
import json
import logging
import cv2

_logger = logging.getLogger(__name__)


class Origin2Coco(object):
    def __init__(self, origin_file_path='../../datasets/lslm/train.txt',
                    origin_image_root='../../datasets/lslm'):
        self.origin_file_path = origin_file_path
        self.origin_image_root = origin_image_root
        self.origin_data = []

        self.images = []
        self.annotations = []
        self.categories = []

        self.image_id = None
        self.label = []

    def load_origin_content(self):
        with open(self.origin_file_path, mode='r', encoding='utf-8') as fp:
            for line in fp.readlines():
                single_image_dict = {}
                line_list = line.split('\t')
                if '\n' in line_list:
                    line_list.remove('\n')
                single_image_dict['file_name'] = line_list[0].strip('')

                single_image_objects = []
                for single_object in line_list[2:]:
                    # print(single_object, type(single_object))
                    single_object_dict = json.loads(single_object.strip('').strip('\n'))
                    single_image_objects.append(single_object_dict)
                single_image_dict['objects'] = single_image_objects

                self.origin_data.append(single_image_dict)

    def generate_coco_format(self):
        coco_info = {}
        info = {"description": "lslm Dataset",
                "version": "1.0",
                 "year": 2020,
                "date_created": "2020/03/27"}
        coco_info['info'] = info

        ann_id = 1
        for single_image in self.origin_data:
            image_file_name = single_image.get('file_name')
            if os.path.isfile(os.path.join(self.origin_image_root, image_file_name)):
                # 添加 image 信息
                self.images.append(self._get_images(image_file_name))
                for single_object in single_image.get('objects'):
                    current_object_label = single_object['value']
                    if current_object_label not in self.label:
                        # 添加 label 信息
                        self.categories.append(self._get_category(current_object_label))
                        self.label.append(current_object_label)

                    self.annotations.append(self._get_annotation(single_object, ann_id))
                    ann_id += 1

        coco_info['images'] = self.images
        coco_info['categories'] = self.categories
        coco_info['annotations'] = self.annotations

        return coco_info

    def _get_images(self, image_file_name):
        image_info = {}
        image_data = cv2.imread(os.path.join(self.origin_image_root, image_file_name))
        height, width, channel = image_data.shape
        self.image_id = image_file_name.split('.')[0]

        # 释放内存
        image_data = None
        image_info['file_name'] = image_file_name
        image_info['width'] = width
        image_info['height'] = height
        image_info['id'] = self.image_id

        return image_info

    def _get_category(self, current_label):
        category = {}
        category['supercategory'] = current_label
        category['name'] = current_label
        # 0 默认为BG
        category['id'] = len(self.label) + 1

        return category

    def _get_annotation(self, single_object, ann_id):
        annotation = {}
        x, y = single_object.get('coordinate')[0]
        # 右下x坐标-左上x坐标
        box_width = single_object.get('coordinate')[1][0] - x
        box_height = single_object.get('coordinate')[1][1] - y
        area = box_width * box_height

        annotation['id'] = ann_id
        annotation['image_id'] = self.image_id
        annotation['category_id'] = self._get_object_category_id(single_object.get('value'))
        annotation['bbox'] = [x, y, box_width, box_height]
        annotation['area'] = area
        # 目标检测数据集，没有实例分割标签（即没有多边形框）
        annotation['segmentation'] = None
        annotation['iscrowd'] = 0

        return annotation

    def _get_object_category_id(self, label):
        for category in self.categories:
            if label == category.get('name'):
                return category.get('id')

        return -1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-OLF", "--originLableFile", dest="originLableFile",
                        default='../../datasets/lslm/train.txt',
                        type=str, help="原始图片标签文件路径.")
    parser.add_argument("-IP", "--imagePath", dest="imagePath",
                        default="../../datasets/lslm",
                        type=str, help="图片路径.")
    parser.add_argument("-OPCJF", "--outputCocoJsonFile", dest="outputCocoJsonFile",
                        default="./lslm_train.json",
                        type=str, help="coco 格式图片标签输出文件路径.")
    args = parser.parse_args()

    pt = Origin2Coco(origin_file_path=args.originLableFile,
                     origin_image_root=args.imagePath)
    pt.load_origin_content()
    coco_info = pt.generate_coco_format()

    with open(args.outputCocoJsonFile, mode='w', encoding='utf-8') as fw:
        json.dump(coco_info, fw)
