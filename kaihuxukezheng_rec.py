import cv2
import re
from datetime import datetime
import numpy as np

from common.json_parse import jsonfile_to_dict
from common.params import args
from common.regular_matching import Regular_match
from ocr_system_base import OCR, text_sys
from common.contour_detection import contour_detection
from common.ocr_utils import re_map, arrimg2string
from template.structured_common import structured, re_map_template
import copy
from loguru import logger as log
import itertools


class InvoiceRec():
    def __init__(self, ori_img):
        model_img_file = r'test/1/317.jpg'
        self.model_img = cv2.imdecode(np.fromfile(model_img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.ori_img = ori_img

    def ocr1(self):
        '''
        使用透视变换前的图片和模板进行ocr文件检测+文本识别模型
        返回开户许可证检测框四个顶点坐标
        '''
        corner_point_model_dict = {}
        ocr = OCR(text_sys, self.model_img)
        json_info, draw_img = ocr(union=False)
        for i in json_info:
            if i['text'].startswith('核准号'):
                corner_point_model_dict['核准号'] = i['quadrangle'][0]
            if i['text'].startswith('编号'):
                corner_point_model_dict['编号'] = i['quadrangle'][0]
            if i['text'].startswith('经审核'):
                corner_point_model_dict['经审核'] = i['quadrangle'][0]
            if i['text'].startswith('符合'):
                corner_point_model_dict['符合'] = i['quadrangle'][0]
            if i['text'].startswith('开立'):
                corner_point_model_dict['开立'] = i['quadrangle'][0]
            if i['text'].startswith('法定代表人'):
                corner_point_model_dict['法定代表人'] = i['quadrangle'][0]
            if i['text'].startswith('开户银行'):
                corner_point_model_dict['开户银行'] = i['quadrangle'][0]
            if i['text'].startswith('账号'):
                corner_point_model_dict['账号'] = i['quadrangle'][0]

        corner_point_ori_dict = {}
        ocr = OCR(text_sys, self.ori_img)
        json_info, draw_img = ocr(union=False)
        for i in json_info:
            if i['text'].startswith('核准号'):
                corner_point_ori_dict['核准号'] = i['quadrangle'][0]
            if i['text'].startswith('编号'):
                corner_point_ori_dict['编号'] = i['quadrangle'][0]
            if i['text'].startswith('经审核'):
                corner_point_ori_dict['经审核'] = i['quadrangle'][0]
            if i['text'].startswith('符合'):
                corner_point_ori_dict['符合'] = i['quadrangle'][0]
            if i['text'].startswith('开立'):
                corner_point_ori_dict['开立'] = i['quadrangle'][0]
            if i['text'].startswith('法定代表人'):
                corner_point_ori_dict['法定代表人'] = i['quadrangle'][0]
            if i['text'].startswith('开户银行'):
                corner_point_ori_dict['开户银行'] = i['quadrangle'][0]
            if i['text'].startswith('账号'):
                corner_point_ori_dict['账号'] = i['quadrangle'][0]

        corner_point_model = []
        corner_point_ori = []
        if '核准号' in corner_point_model_dict and '核准号' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['核准号'])
            corner_point_ori.append(corner_point_ori_dict['核准号'])
        if '编号' in corner_point_model_dict and '编号' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['编号'])
            corner_point_ori.append(corner_point_ori_dict['编号'])
        if '经审核' in corner_point_model_dict and '经审核' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['经审核'])
            corner_point_ori.append(corner_point_ori_dict['经审核'])
        if '符合' in corner_point_model_dict and '符合' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['符合'])
            corner_point_ori.append(corner_point_ori_dict['符合'])
        if '开立' in corner_point_model_dict and '开立' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['开立'])
            corner_point_ori.append(corner_point_ori_dict['开立'])
        if '法定代表人' in corner_point_model_dict and '法定代表人' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['法定代表人'])
            corner_point_ori.append(corner_point_ori_dict['法定代表人'])
        if '开户银行' in corner_point_model_dict and '开户银行' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['开户银行'])
            corner_point_ori.append(corner_point_ori_dict['开户银行'])
        if '账号' in corner_point_model_dict and '账号' in corner_point_ori_dict:
            corner_point_model.append(corner_point_model_dict['账号'])
            corner_point_ori.append(corner_point_ori_dict['账号'])

        return corner_point_model[:2]+corner_point_model[-2:], corner_point_ori[:2]+corner_point_ori[-2:]
        #return corner_point_model[:4], corner_point_ori[:4]
        #return corner_point_model, corner_point_ori

    def combination(self, corner_point_model, corner_point_ori):
        '''
        从已知的点中进行四点排列组合
        '''
        corner_point_model_zuhe = list(map(list, list(itertools.combinations(corner_point_model, 4))))
        corner_point_ori_zuhe = list(map(list, list(itertools.combinations(corner_point_ori, 4))))
        model_ori = []
        for i in range(len(corner_point_model_zuhe)):
            combined_point = [corner_point_model_zuhe[i], corner_point_ori_zuhe[i]]
            model_ori.append(combined_point)
        return model_ori

    '''
    #评价透视变换好坏选出最合适的一组点
    def evaluate_perspective_transform(self, model_ori):

        def calculate_similarity(image1, image2):

            # Convert the images to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # Resize the images to the same size
            resized1 = cv2.resize(gray1, (300, 300))
            resized2 = cv2.resize(gray2, (300, 300))

            # Calculate the structural similarity index (SSIM) between the two images
            ssim = cv2.compare_ssim(resized1, resized2)

            return ssim

        ssim_list = []
        for i in model_ori:
            img = self.ori_img
            b, g, r = list(map(int, cv2.meanStdDev(img)[0].squeeze()))
            borderValue = (b, g, r)
            corner_point_model = np.float32(i[0])
            corner_point_ori = np.float32(i[1])
            M = cv2.getPerspectiveTransform(corner_point_ori, corner_point_model)
            new_img = cv2.warpPerspective(img, M, (1826, 1280), borderValue=borderValue)
            ssim = calculate_similarity(img, new_img)
            ssim_list.append(ssim)

        return model_ori[ssim_list.index(max(ssim_list))]
    
    '''

    def evaluate_perspective_transform(self, model_ori):
        '''评价透视变换质量，选择与模板最相似的一组变换参数'''

        def calculate_similarity(transformed_img, template_img):
            """计算变换后图像与模板图像的结构相似性"""
            # 统一转换为灰度图
            gray_trans = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

            # 调整到相同尺寸（取模板尺寸）
            h, w = gray_template.shape[:2]
            resized_trans = cv2.resize(gray_trans, (w, h))

            # 计算结构相似性
            return cv2.compare_ssim(resized_trans, gray_template)

        ssim_list = []
        for model_points, ori_points in model_ori:
            # 执行透视变换
            M = cv2.getPerspectiveTransform(
                np.float32(ori_points),  # 原始图像点
                np.float32(model_points)  # 模板图像点
            )
            warped_img = cv2.warpPerspective(
                self.ori_img, M,
                (self.model_img.shape[1], self.model_img.shape[0]),  # 使用模板的尺寸
                borderValue=(255, 255, 255)  # 白色填充
            )

            # 计算与模板的相似度
            ssim = calculate_similarity(warped_img, self.model_img)
            ssim_list.append(ssim)

        # 返回相似度最高的组合
        best_idx = np.argmax(ssim_list)
        return model_ori[best_idx]

    def invoice_warp(self, corner_point_model, corner_point_ori):
        '''
        根据开户许可证检测框四点坐标进行透视变换
        '''
        img = self.ori_img
        b, g, r = list(map(int, cv2.meanStdDev(img)[0].squeeze()))
        borderValue = (b, g, r)

        if corner_point_ori is not None:
            #p2 = np.array([(78, 136), (1453, 136), (1453, 1001), (78, 1001)], dtype=np.float32)
            corner_point_model = np.float32(corner_point_model)
            corner_point_ori = np.float32(corner_point_ori)
            M = cv2.getPerspectiveTransform(corner_point_ori, corner_point_model)
            new_img = cv2.warpPerspective(img, M, (1826, 1280), borderValue=borderValue)
            self.warp = True
        else:
            new_img = img
            self.warp = False
        if args.is_visualize:
            cv2.imwrite(r'test/adjust.jpg', new_img)
        return new_img

    def ocr2(self, new_img):
        '''
        使用透视变换后的图片进行ocr文件检测+文本识别模型
        '''
        ocr = OCR(text_sys, new_img)
        return ocr

    def make_template(self, new_img, ocr):
        '''
        根据营业执照轮廓长宽比选择相应的json文件做模板
        '''
        json_info, draw_img = ocr(union=False)
        #json_info_union, draw_img = ocr(union = True, max_x_dist = 50, min_y_overlap_ratio = 0.5)

        rectangle_dict, drop_ind, key_ind = structured(labelme_file=r'317.json')

        # 做模板
        drop_ind += [0]
        tmp = np.zeros((1280, 1826), dtype = 'uint8')

        for r in rectangle_dict:
            cv2.rectangle(tmp, tuple(rectangle_dict[r][0]), tuple(rectangle_dict[r][1]), r, thickness = cv2.FILLED)

        if args.is_visualize:
            anchor_img = copy.deepcopy(new_img)
            for r in rectangle_dict:
                cv2.rectangle(anchor_img, tuple(rectangle_dict[r][0]), tuple(rectangle_dict[r][1]), 255,
                              thickness = 2)
            cv2.imwrite('./test/draw_anchor_box.jpg', anchor_img)

        key_ind = copy.deepcopy(key_ind)
        for key in key_ind:
            key_ind[key][2] = rectangle_dict[key]  # todo

        for t in json_info:
            point = [int(t["bbox"][0] + t["bbox"][2]) // 2, int(t["bbox"][1] + t["bbox"][3]) // 2]
            text = t["text"]
            try:
                label_ind = tmp[point[1]][point[0]]
            except:
                log.warning(f'index {label_ind} is out of bounds')

            if label_ind in drop_ind:
                continue
            else:
                key_ind[label_ind][1] += text

        return key_ind, draw_img

    def get_result(self, key_ind):
        #flag_code_res, invoice_type, invoice_daima, invoice_haoma, total_money, date, check_code = self.code_res(new_img)
        #coderes = {"发票抬头": invoice_type, "发票代码": invoice_daima, "发票号码": invoice_haoma,
                   #"开票日期": date, "合计金额": total_money, '校验码': check_code}
        def remove_letters_and_punctuation(text):
            # 去除多余的字母标点符号以及空格
            cleaned_text = re.sub('[a-zA-Z（）() ]', '', text)
            return cleaned_text

        def remove_chinese_and_punctuation(text):
            # 去除多余的汉字和标点以及空格
            pattern = re.compile(r'[\u4e00-\u9fa5，。！？；：:“”‘’【】《》（）【】、——…… ]')
            result = re.sub(pattern, '', text)
            return result

        def remove_numbers_and_letters(string):
            # 去除多余的字母和数字
            pattern = r'[a-zA-Z0-9]'
            result = re.sub(pattern, '', string)
            return result

        def replace_first_character(string):
            # 如果首字符是1开头则替换首字符为J
            if string[0] == '1':
                return 'J' + string[1:]
            else:
                return string

        res = []
        for k in key_ind:
            res.append({"key": key_ind[k][0], "value": key_ind[k][1]})

        result = {}
        for dic in res:
            result[dic['key']] = dic['value']
            if dic['key'] == '核准号':
                result[dic['key']] = remove_chinese_and_punctuation(dic['value'])
                result[dic['key']] = replace_first_character(result[dic['key']])

            if dic['key'] == '编号':
                result[dic['key']] = remove_chinese_and_punctuation(dic['value'])
            if dic['key'] == '公司':
                if dic['value'].endswith('公'):
                    result[dic['key']] = dic['value'] + '司'
                try:
                    result[dic['key']] = result[dic['key']].split('，')[1]
                except:
                    pass
            if dic['key'] == '开户银行':
                result[dic['key']] = remove_numbers_and_letters(dic['value'])
                result[dic['key']] = result[dic['key']].replace('开户银行', '')
                if result[dic['key']].endswith('支'):
                    result[dic['key']] = result[dic['key']] + '行'

            if dic['key'] == '法定代表人':
                result[dic['key']] = remove_letters_and_punctuation(dic['value']).replace('法定代表人', '').replace('单位负责人', '')
            if dic['key'] == '账号':
                result[dic['key']] = remove_chinese_and_punctuation(dic['value'])

        return result

    def filter_result(self, result, key_list):
        if key_list == []:
            return result

        finally_result = dict(zip(key_list, [""]*len(key_list)))
        for k, v in result.items():
            if k in key_list:
                finally_result[k] = v
        return finally_result


    def split(self, draw_img):
        size_img = draw_img.shape[:2]
        half_width = int(size_img[1] // 2)
        draw_img_left = draw_img[:, :half_width]
        draw_img_right = draw_img[:,- half_width:]
        return draw_img_left, draw_img_right

    def __call__(self, key_list=[]):
        corner_point_model, corner_point_ori = self.ocr1()
        #model_ori = self.combination(corner_point_model, corner_point_ori)
        #model_ori_best = self.evaluate_perspective_transform(model_ori)
        new_img = self.invoice_warp(corner_point_model, corner_point_ori)
        ocr = self.ocr2(new_img)

        if self.warp:
            key_ind, draw_img = self.make_template(new_img, ocr)
            result = self.get_result(key_ind)
        else:
            log.warning('轮廓检测失败')
            json_path = r'template/config.json'
            regulation_key = jsonfile_to_dict(json_path = json_path)
            rm = Regular_match(ocr, None, regulation_key, shape = new_img.shape)
            result, draw_img = rm()

        
        # finally_result = self.filter_result(result, key_list = key_list)

        #draw_img_left, draw_img_right = self.split(draw_img)
        #result['draw_img_left'] = arrimg2string(draw_img_left)
        #result['draw_img_right'] = arrimg2string(draw_img_right)

        return result

