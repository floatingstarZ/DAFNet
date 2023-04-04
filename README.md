<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>


[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)


  <img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>


[📘Documentation](https://mmdetection.readthedocs.io/en/v2.18.1/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/v2.18.1/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/zh_CN/v2.18.1/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/v2.18.1/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

## DAFNet: Introduction
The repository is a code implementation of DAFNet.
Since the dataset is private, we only provide the code.
We provide an implementation based on Faster R-CNN, 
and other detection methods can be implemented with minor changes.

The configurations of all fusion methods can be found in 
./DAFNet_configs

where:
./DAFNet_configs/DAFNet_faster_rcnn_AFF.py is our DAFNet 

## DAFNet: Installation

You can prepare the environment by following steps: 

conda create -n DAFNet python=3.9 -y

conda activate DAFNet

pip install torch==1.10.1+cu102

pip install torchvision==0.11.2

pip install -r requirements/build.txt

pip install mmcv-full

pip install tifffile

python setup.py develop

## DAFNet: Usage
Please follow the guidance of MMDetection

## Citation

If you use this toolbox or benchmark in your research, please cite this project.
The citation of DAFNet will be provided soon!!

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

