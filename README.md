# eegcls

`eegcls` 是一个面向 OpenBCI `txt` 文件的离线 EEG 分类最小产品库。

当前版本已经打通了这条主链路：

- 读取 OpenBCI `txt`
- 提取前 `8` 个 EEG 通道
- 按固定窗口切片
- 对每个窗口做标准化
- 使用 `EEGNet` 训练三分类模型
- 保存可复用的模型 artifact
- 对单个 `txt` 文件做逐窗口推理，并按时间顺序输出结果

当前版本的目标不是做通用实验平台，而是先把“可训练、可保存、可推理”的产品链路跑通。

## 1. 当前能力

目前已经实现的能力如下：

- 固定输入格式：OpenBCI `txt`
- 固定通道数：只取前 `8` 个 EEG 通道
- 固定任务类型：离线、单标签分类
- 固定推理形式：输入一个 `txt`，输出按时间顺序排列的逐窗口分类结果
- 固定模型：当前只接入 `EEGNet`

## 2. 当前目录说明

主要代码目录：

- [eegcls/openbci.py](/home/xyr/workspace/eegcls/eegcls/openbci.py)：OpenBCI `txt` 解析
- [eegcls/preprocess.py](/home/xyr/workspace/eegcls/eegcls/preprocess.py)：窗口切片与标准化
- [eegcls/dataset.py](/home/xyr/workspace/eegcls/eegcls/dataset.py)：数据集扫描与窗口样本构建
- [eegcls/modeling.py](/home/xyr/workspace/eegcls/eegcls/modeling.py)：模型构建，当前只支持 `EEGNet`
- [eegcls/training.py](/home/xyr/workspace/eegcls/eegcls/training.py)：训练与评估
- [eegcls/artifact.py](/home/xyr/workspace/eegcls/eegcls/artifact.py)：模型 artifact 保存与加载
- [eegcls/inference.py](/home/xyr/workspace/eegcls/eegcls/inference.py)：单文件推理

脚本目录：

- [scripts/build_toy_dataset.py](/home/xyr/workspace/eegcls/scripts/build_toy_dataset.py)：用一个原始 `txt` 构造 toy 三分类数据集
- [scripts/train_eegnet.py](/home/xyr/workspace/eegcls/scripts/train_eegnet.py)：训练 `EEGNet`
- [scripts/predict_file.py](/home/xyr/workspace/eegcls/scripts/predict_file.py)：对单个 `txt` 做推理

## 3. 运行环境

当前代码默认使用：

- Python 3
- `numpy`
- `torch`

说明：

- 当前环境里没有安装 `PyYAML`，所以现在 artifact 和配置文件使用的是 `json`
- 当前训练与推理脚本都通过 `PYTHONPATH=.` 运行

## 4. 数据格式约定

### 4.1 输入文件

输入必须是 OpenBCI 导出的 `txt` 文件。

当前解析逻辑假设：

- 第一行是表头
- 表头前 8 列对应 `EXG Channel 0` 到 `EXG Channel 7`
- 实际建模时只读取前 8 列 EEG 数据

解析后的内部张量形状为：

```text
[C, T]
```

其中：

- `C = 8`
- `T = 时间点数`

### 4.2 训练数据集目录

正式训练时，数据目录应组织为：

```text
dataset/
  train/
    class0/
      *.txt
    class1/
      *.txt
    class2/
      *.txt
  val/
    class0/
      *.txt
    class1/
      *.txt
    class2/
      *.txt
  test/
    class0/
      *.txt
    class1/
      *.txt
    class2/
      *.txt
```

规则如下：

- 标签来自子目录名
- 一个 `txt` 只能属于一个 split
- 一个 `txt` 文件切出来的所有窗口都继承该文件所在目录的标签

## 5. 预处理与窗口切片

当前版本预处理比较克制，只做最小必需步骤：

- 提取前 8 个通道
- 固定窗口切片
- 每个窗口单独做标准化

默认训练参数里使用：

- `sampling_rate = 250`
- `window_size = 128`
- `stride = 64`

这意味着每个窗口的内部形状固定为：

```text
[8, 128]
```

## 6. 快速开始

### 6.1 用一个原始 txt 构造 toy 数据集

这个项目当前自带了一个最小演示流程，可以先用一个原始 `txt` 造一个三分类 toy 数据集，只用于验证工程链路是否跑通。

命令：

```bash
python scripts/build_toy_dataset.py \
  --source OpenBCI-RAW-2026-04-12_20-22-01.txt \
  --output demo_dataset
```

执行后会生成：

```text
demo_dataset/
  train/
  val/
  test/
```

每个 split 下会生成 `class0`、`class1`、`class2` 三个类别目录。

说明：

- 这个 toy 数据集是从同一个原始文件按顺序切成多个片段构造出来的
- 它的作用只是验证代码链路，不代表真实可用的训练数据

### 6.2 训练 EEGNet

命令：

```bash
PYTHONPATH=. python scripts/train_eegnet.py \
  --dataset-root demo_dataset \
  --artifact-dir artifacts/eegnet_demo \
  --epochs 6 \
  --batch-size 8 \
  --window-size 128 \
  --stride 64
```

训练完成后，会在 `artifacts/eegnet_demo/` 下面生成模型 artifact。

## 7. Artifact 说明

当前 artifact 目录包含：

- `model.pt`：模型权重
- `model_config.json`：模型配置
- `preprocess_config.json`：预处理配置
- `label_map.json`：标签映射
- `train_summary.json`：训练摘要
- `library_version.txt`：当前库版本

这些文件共同决定了后续推理行为。

## 8. 单文件推理

训练完成后，可以对一个 OpenBCI `txt` 文件做逐窗口推理。

命令：

```bash
PYTHONPATH=. python scripts/predict_file.py \
  --artifact-dir artifacts/eegnet_demo \
  --input OpenBCI-RAW-2026-04-12_20-22-01.txt
```

输出是一个 JSON 列表，每个元素代表一个窗口样本，按时间顺序排列。

每条结果当前包含：

- `window_index`
- `start_idx`
- `end_idx`
- `start_time`
- `end_time`
- `pred_index`
- `pred_label`
- `confidence`
- `probabilities`

## 9. Python 中调用

除了脚本方式，也可以直接在 Python 中调用：

```python
from eegcls.inference import predict_file

results = predict_file(
    artifact_dir="artifacts/eegnet_demo",
    txt_path="OpenBCI-RAW-2026-04-12_20-22-01.txt",
)

print(results[0])
```

## 10. 当前限制

当前版本只是最小可运行版本，限制比较明确：

- 目前只支持 `EEGNet`
- 目前只支持 OpenBCI `txt`
- 目前只读取前 `8` 个 EEG 通道
- 目前只做离线推理，不做实时流式推理
- 目前预处理只包含窗口切片和标准化，还没有加入更完整的滤波流程
- 当前 demo 使用的 toy 数据集不具备实际训练价值，只适合验证工程流程

## 11. 推荐的下一步

如果要把这个库继续往可用版本推进，建议按这个顺序做：

1. 接入真实的 `train/val/test/<label>/*.txt` 数据集
2. 补全 OpenBCI `txt` 解析校验与异常处理
3. 增加更正式的配置文件结构
4. 在 `preprocess` 中加入 notch/band-pass/resample 等信号处理
5. 补充更完整的评估指标和结果导出
6. 再考虑接入更多模型，而不是一开始就并行支持很多 backbone
