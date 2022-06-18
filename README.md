# Towards Principled Disentanglement for Domain Generalization, CVPR, 2022 (Oral)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DDG is a PyTorch implementation of [Towards Principled Disentanglement for Domain Generalization](https://arxiv.org/abs/2111.13839) based on [DomainBed](https://github.com/facebookresearch/DomainBed).
## Available datasets

The [currently available datasets](datasets.py) are:

* RotatedMNIST ([Ghifary et al., 2015](https://arxiv.org/abs/1508.07680))
* VLCS  ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* WILDS ([Koh et al., 2020](https://arxiv.org/abs/2012.07421)) Camelyon17 ([Bandi et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30716025/)) about tumor detection in tissues

Send us a PR to add your dataset! Any custom image dataset with folder structure `dataset/domain/class/image.xyz` is readily usable. While we include some datasets from the [WILDS project](https://wilds.stanford.edu/), please use their [official code](https://github.com/p-lambda/wilds/) if you wish to participate in their leaderboard.

## Available model selection criteria

[Model selection criteria](model_selection.py) differ in what data is used to choose the best hyper-parameters for a given model:

* `IIDAccuracySelectionMethod`: A random subset from the data of the training domains.
* `LeaveOneOutSelectionMethod`: A random subset from the data of a held-out (not training, not testing) domain.
* `OracleSelectionMethod`: A random subset from the data of the test domain.

## Quick start

Download the datasets:

```python
python scripts/download.py \
       --data-dir /my/datasets/path
```

Train a model:

```python
python train.py\
       --data-dir /my/datasets/path\
       --algorithm ERM\
       --dataset RotatedMNIST
```

Pretrain the decoder in DDG model:

```python
python train.py\
       --data-dir /my/datasets/path\
       --algorithm DDG\
       --dataset PACS\
       --stage 0
```

Train the DDG model with pretrained decoder:

```python
python train.py\
       --data-dir /my/datasets/path\
       --algorithm DDG\
       --gen-dir /my/models/model.pkl
       --dataset PACS\
       --stage 1
```

### Citation 
If you find this repo useful, please consider citing: 
```
@inproceedings{zhang2022DDG,
  title={Towards principled disentanglement for domain generalization},
  author={Zhang, Hanlin and Zhang, Yi-Fan and Liu, Weiyang and Weller, Adrian and Sch{\"o}lkopf, Bernhard and Xing, Eric P},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8024--8034},
  year={2022}
}
```
