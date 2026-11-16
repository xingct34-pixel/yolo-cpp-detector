# YOLO 检测部署 | tools/validate_coco8.py
# 一次小样本回归的入口。它把已有数据集的标签换算成原图坐标，分别调用
# yolo_coco_predict 导出两个模型的预测，再用项目评估器和官方 COCOeval 交叉核对。
# 先看 main 里的文件流向，再看 convert 如何把归一化框换成像素。
#
# 阅读地图（按代码顺序）：
#   模块 0 准备工作  -> 导入工具、设置路径
#   模块 1 convert   -> 把 YOLO 标注翻译成 COCO 标注
#   模块 2 official_ap -> 用官方工具算"标准答案"分数
#   模块 3 main      -> 串起整个验收流程
#   模块 4 入口      -> 直接运行文件时启动 main


# 模块 0：准备工作（导入与路径设置）
# 功能：引入后面要用的标准库、第三方库和项目内部工具，并告诉 Python 去哪里找项目代码。
# 目的：让本脚本能直接运行，并且能复用项目里已有的预处理、哈希、评估函数，
#       保证这里的处理方式和量化时完全一致。


import argparse  # 解析命令行参数（--dataset 之类）
import contextlib  # 临时改变输出去向，用来屏蔽官方工具的啰嗦打印
import hashlib  # 计算 SHA256 指纹
import importlib.metadata  # 查询已安装库的版本号，写进报告方便复现
import io  # 提供"内存里的假文件"，用来接住被屏蔽的输出
import json  # 读写 JSON 文件
from pathlib import Path  # 用对象的方式处理文件路径，比字符串拼接安全
import subprocess  # 在 Python 里启动外部程序（这里用来启动 C++ 预测程序）
import sys  # 用来修改模块搜索路径

import cv2  # OpenCV，用来读图片
from pycocotools.coco import COCO  # 官方 COCO 数据集对象
from pycocotools.cocoeval import COCOeval  # 官方 COCO 评分器

# 把项目根目录加入模块搜索路径，这样才能导入上一级目录里的 tools 包
# （本文件在 tools/ 里，parents[1] 就是 tools 的上一级，也就是项目根目录）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# prepare：图片预处理（与量化时一致）；sha256：文件哈希；evaluate：项目自己写的 AP 评估函数
from tools.cpu_quantize import prepare, sha256
from tools.evaluate_detection import evaluate


# 模块 1：格式转换（convert）
# 功能：把外部准备好的 COCO8 验证集（4 张图片 + YOLO 格式标注）转成 COCO 标注格式，
#       同时给每张图算一个"预处理后内容的指纹"。只读本地文件，不下载、不伪造真值。
# 目的：评估工具只认 COCO 格式，所以必须先翻译；指纹则用来在后面证明
#       "考试用的图"和"量化校准用的图"没有内容完全相同的张量。
# 通俗讲：YOLO 和 COCO 是两种"标注记账方式"，就像同一笔账，一个用"中心点+比例"记，
#         一个用"左上角+像素"记，这里负责把账本重新抄一遍。


def convert(dataset, names):
    # images：图片信息列表；annotations：标注框列表；hashes：每张图预处理后的指纹
    images, annotations, hashes = [], [], {}
    # sorted 保证图片顺序固定，每次运行得到的编号一致；编号从 1 开始
    for index, path in enumerate(sorted((dataset / 'images/val').glob('*.jpg')), 1):
        image = cv2.imread(str(path))
        # 读不出来（文件损坏等）就直接报错，不能悄悄跳过
        if image is None:
            raise ValueError('cannot decode validation image')
        # image.shape 是 (高, 宽, 通道数)，这里只取前两个
        height, width = image.shape[:2]
        images.append({'id': index, 'file_name': path.name, 'width': width, 'height': height})
        # 计算"预处理后张量"的哈希。后面用它和校准数据的哈希对比，
        # 目的是确认评估用的图和量化校准用的图没有完全相同的预处理张量；近似画面仍可能非常相似
        hashes[path.name] = hashlib.sha256(prepare(image).tobytes()).hexdigest()
        # 读取同名的 .txt 标注文件，一行一个目标
        for line in (
            (dataset / 'labels/val' / path.with_suffix('.txt').name).read_text().splitlines()
        ):
            # YOLO 每行格式：类别 中心x 中心y 宽 高，后四个都是 0~1 的相对比例
            category, x, y, w, h = map(float, line.split())
            # 合法性检查：类别必须是整数且在类别表范围内，四个坐标必须在 0~1 之间
            if (
                not category.is_integer()
                or not 0 <= category < len(names)
                or not all(0 <= z <= 1 for z in (x, y, w, h))
            ):
                raise ValueError('invalid external YOLO label')
            # 转换坐标：YOLO 是"中心点 + 宽高（相对比例）"，
            # COCO 是"左上角 x、y + 宽高（像素）"。
            # 先用中心点减去半个宽/高得到左上角，再乘以图片宽高换成像素
            # 例：图宽 640，x=0.5，w=0.2 -> 左上角 x = (0.5-0.1)*640 = 256，宽 = 128 像素
            box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
            annotations.append(
                {
                    'id': len(annotations) + 1,  # 标注框全局编号，从 1 开始
                    'image_id': index,  # 这个框属于哪张图
                    'category_id': int(category) + 1,  # YOLO 类别从 0 起，COCO 这里从 1 起，所以加 1
                    'bbox': box,  # 框的位置：[左上角x, 左上角y, 宽, 高]，单位像素
                    'area': box[2] * box[3],  # 面积 = 宽 × 高，COCO 评估时用来区分大中小目标
                    'iscrowd': 0,  # 0 表示不是"人群"类密集标注，当作普通单个目标
                }
            )

    # 数据集规模校验：必须正好 4 张图，防止拿错数据集
    if len(images) != 4:
        raise ValueError('expected exactly four COCO8 val images')
    # 返回两样东西：COCO 格式的完整标注字典，以及每张图的指纹
    return {
        'info': {
            # 说明数据来源：这 4 张图出自 COCO train2017，不是独立的测试集
            'description': 'COCO8 val smoke subset from COCO train2017; not held-out full COCO'
        },
        'images': images,
        'annotations': annotations,
        # 类别表：编号从 1 开始，对应上面 category_id 的 +1
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(names)],
    }, hashes


# 模块 2：官方复核（official_ap）
# 功能：用固定版本的官方 pycocotools 对同一份预测结果计算 AP（检测精度分数）。
# 目的：作为"标准答案"，用来检验项目自己写的评估函数算得对不对，
#       防止因为自研评分程序有 bug 而得出错误结论。
# 通俗讲：自己写的评分程序可能有 bug，所以再找官方评分程序算一遍，两边分数一致才放心。


def official_ap(dataset, predictions):
    # 没有任何预测框时官方工具会报错，这种情况直接定义为 AP 全为 0
    if not predictions:
        return {'AP50': 0.0, 'AP75': 0.0, 'mAP50_95': 0.0}
    # pycocotools 会往终端打印大量过程信息，这里把标准输出丢到内存里，保持终端干净
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        # 直接把内存里的字典当作真值数据集，不需要写文件再读
        coco.dataset = dataset
        coco.createIndex()  # 建立索引，之后才能按图片、类别查询
        result = coco.loadRes(predictions)  # 加载预测结果
        evaluator = COCOeval(coco, result, 'bbox')  # 'bbox' 表示按检测框评估
        evaluator.evaluate()  # 逐图逐类别匹配预测框与真值框
        evaluator.accumulate()  # 汇总出精度-召回曲线
        evaluator.summarize()  # 计算最终指标，结果存在 evaluator.stats 里
    # stats[0]：IoU 从 0.5 到 0.95 的平均 AP（最常引用的 mAP）
    # stats[1]：IoU=0.5 时的 AP（要求宽松：交集面积至少达到并集面积的一半才满足这个 IoU 门槛）
    # stats[2]：IoU=0.75 时的 AP（要求严格：框要贴得很准才算对）
    return {
        'mAP50_95': float(evaluator.stats[0]),
        'AP50': float(evaluator.stats[1]),
        'AP75': float(evaluator.stats[2]),
    }


# 模块 3：完整回归流程（main）
# 功能：串起整个测试：转换标注 -> 校验数据和模型身份 -> 用 C++ 程序分别跑 FP32 和 INT8 模型出预测 ->
#       计算 AP 并和官方结果交叉验证 -> 汇总 INT8 相对 FP32 的精度变化，写入 report.json。
# 目的：回答"模型量化之后，检测效果到底掉了多少"，并且保证这个回答是可信的
#       （图不重复、模型没拿错、评分程序没算错）。
# 通俗讲：这是"量化之后检测效果到底掉了多少"的小型验收考试，
#         main 就是监考老师，负责出卷、收卷、判卷、核对答案、出成绩单。


def main():
    # ---- 第一步：读取命令行参数 ----
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)  # COCO8 数据集目录
    parser.add_argument('--fp32', type=Path, required=True)  # 原始 FP32 模型（高精度版，当对照组）
    parser.add_argument('--int8', type=Path, required=True)  # 量化后的 INT8 模型（压缩版，被检验对象）
    parser.add_argument('--calibration-report', type=Path, required=True)  # 量化脚本生成的 report.json
    parser.add_argument('--binary', type=Path, default=Path('build-local/yolo_coco_predict'))  # C++ 预测程序
    parser.add_argument('--output', type=Path, required=True)  # 本次实验的输出目录
    args = parser.parse_args()

    # ---- 第二步：准备输出目录 ----
    # 输出目录必须是新的，防止覆盖之前的实验记录
    if args.output.exists():
        parser.error('choose a new output directory to retain prior evidence')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]

    # ---- 第三步：转换标注并保存 ----
    # 读取 80 个类别名称，一行一个
    names = (root / 'data/coco.txt').read_text().splitlines()
    dataset, hashes = convert(args.dataset, names)
    # 把转换好的 COCO 标注存盘，C++ 程序要用它
    annotation = args.output / 'annotations.json'
    annotation.write_text(json.dumps(dataset))
    calibration = json.loads(args.calibration_report.read_text())

    # ---- 第四步：检查独立性，并先搭好报告框架 ----
    # 求交集：评估图的指纹和校准图的指纹有没有重复。
    # 重复就意味着量化时"见过"这些图，评估结果会偏乐观
    overlap = set(hashes.values()) & set(calibration['calibration_tensor_sha256'])
    # 先把报告的基本信息填好，各模型的结果后面逐个补充
    report = {
        'dataset': 'Ultralytics COCO8 val, four images sourced from COCO train2017',
        # 明确声明结论的适用范围：只是小规模流程回归，且模型预训练时可能见过这些图，不能当作总体精度
        'scope': 'small labeled pipeline regression; not independent of model pretraining; no population accuracy claim',
        'pycocotools': importlib.metadata.version('pycocotools'),  # 记录版本，方便复现
        'validation_images': 4,
        'ground_truth_boxes': len(dataset['annotations']),  # 真值框总数
        'calibration_tensor_overlap': len(overlap),  # 与校准图重复的数量，应该是 0
        'validation_tensor_sha256': hashes,
        'annotation_sha256': sha256(annotation),
        'models': {},  # 各模型的结果，下面循环里填
    }

    # ---- 第五步：三道"身份与独立性"关卡，任何一道不过就停止 ----
    # 1. 校准数据与评估数据有重复 -> 不独立，终止
    if overlap:
        raise RuntimeError('calibration and evaluation tensors overlap')
    # 2. 传入的 FP32 模型哈希必须与量化时记录的一致，确认没拿错模型文件
    if sha256(args.fp32) != calibration['source_model_sha256']:
        raise RuntimeError('FP32 identity differs from supplied calibration evidence')
    # 3. 传入的 INT8 模型哈希同理
    if sha256(args.int8) != calibration['quantized_sha256']:
        raise RuntimeError('INT8 identity differs from supplied calibration evidence')

    # ---- 第六步：两个模型各考一遍 ----
    # 两个模型走完全相同的流程，保证对比公平
    for label, model in [('fp32', args.fp32), ('int8', args.int8)]:
        destination = args.output / (label + '-predictions.json')
        # 调用 C++ 原生程序推理，stdout 和 stderr 都写进日志文件留证。
        # 用原生程序而不是 Python 推理，是为了测到真正部署时的整条链路
        with (args.output / (label + '-native.log')).open('w') as log:
            subprocess.run(
                [
                    str(args.binary.resolve()),  # 要运行的 C++ 程序
                    str(annotation.resolve()),  # 参数1：标注文件
                    str((args.dataset / 'images/val').resolve()),  # 参数2：图片目录
                    str(model.resolve()),  # 参数3：要测的模型
                    str(root / 'data/coco.txt'),  # 参数4：类别名称表
                    str(destination.resolve()),  # 参数5：预测结果写到哪
                ],
                stdout=log,
                stderr=log,
                check=True,  # 程序返回非零就抛异常，不会带着失败结果继续往下走
            )
        predictions = json.loads(destination.read_text())['predictions']
        # 项目自己写的评估函数算一遍
        local = evaluate(dataset, predictions)
        # 官方工具再算一遍
        reference = official_ap(dataset, predictions)
        # 逐项求两者差的绝对值
        difference = {name: abs(local[name] - reference[name]) for name in reference}
        # 差异超过百万分之一就认为自研评估有问题，终止，避免用错误的评分程序下结论
        if max(difference.values()) > 1e-6:
            raise RuntimeError('non-crowd evaluator differs from official COCOeval')
        report['models'][label] = {
            'model_sha256': sha256(model),  # 模型指纹，证明测的是哪个文件
            'predictions': len(predictions),  # 预测框总数
            'reference_ap': reference,  # 官方 AP，作为最终采信的结果
            'custom_ap': {k: local[k] for k in reference},  # 自研 AP，仅用于对照
            'reference_difference': difference,  # 两种算法的差距，应该几乎为 0
        }

    # ---- 第七步：算出量化前后的差距并出成绩单 ----
    # 计算 INT8 减 FP32 的 AP 变化量：
    # 负数表示量化后精度下降，接近 0 表示几乎无损，正数则可能只是小样本波动
    report['int8_minus_fp32'] = {
        key: report['models']['int8']['reference_ap'][key]
        - report['models']['fp32']['reference_ap'][key]
        for key in ('AP50', 'AP75', 'mAP50_95')
    }
    # ensure_ascii=False 让中文等非 ASCII 字符原样保存，不转成 \uXXXX
    (args.output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    )
    # 同时打印到终端，方便直接查看
    print(json.dumps(report, ensure_ascii=False, indent=2))


# 模块 4：脚本入口
# 功能：只有直接运行本文件（python validate_coco8.py ...）时才执行 main()。
# 目的：这个文件也可能被别的文件 import 来借用函数，此时不应该自动开始跑整套测试。


if __name__ == '__main__':
    main()
