## Revisiting Adversarial Patches for Designing Camera-Agnostic Attacks against Person Detection (NeurIPS 2024)
[Project](https://camera-agnostic.github.io/) | [Paper](https://nips.cc/virtual/2024/poster/96825)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=weihui1308/CAP)

This repository enables the generation of adversarial patches that remain effective under multiple camera devices.

[Hui Wei](https://weihui1308.github.io/)<sup>1</sup>, [Zhixiang Wang](https://lightchaserx.github.io/)<sup>2</sup>, [Kewei Zhang](https://scholar.google.com/citations?user=cFk7BcAAAAAJ&hl=en)<sup>1</sup>, [Jiaqi Hou]()<sup>1</sup>, [Yuanwei Liu]()<sup>1</sup>, [Hao Tang](https://ha0tang.github.io/)<sup>3</sup>, [Zheng Wang](https://wangzwhu.github.io/home/)<sup>1</sup>

<sup>1</sup>Wuhan University,&nbsp;&nbsp;<sup>2</sup>University of Tokyo,&nbsp;&nbsp;<sup>3</sup>Peking University. 

### Update

- **2024.10.10**: Repo is released.
- **2024.09.26**: Paper is accepted to NeurIPS 2024.

### :toolbox: Setup
1. Clone this repo:
```bash
git clone https://github.com/weihui1308/CAP.git
cd CAP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the weight files:
- Finetuned YOLOv5 model: [finetune_yolov5s_onINRIA.pt](https://github.com/weihui1308/CAP/tree/main/assets/checkpoints)
- Camera ISP proxy network model: [checkpoint_ISPNet.pth](https://drive.google.com/file/d/1k9g42kr67ygfGAaPcyl6cVLaYNM-l30q/view?usp=sharing)

### :rainbow: Training and Evaluation
1. Optimize the adversarial patch:
```bash
python train.py --data config/data_config.yaml --weights checkpoints/finetune_yolov5s_onINRIA.pt --batch_size 32 --epochs 1000
```

2. Evaluate a given adversarial patch:
```bash
python val.py --patch_name patch/onePatch.png
```


### Citation

If you find our work useful, please kindly cite as:
```
@inproceedings{wei2024cap,
      title={Revisiting Adversarial Patches for Designing Camera-Agnostic Attacks against Person Detection},
      author={Wei, Hui and Wang, Zhixiang and Zhang, Kewei and Hou, Jiaqi and Liu, Yuanwei and Tang, Hao and Wang, Zheng},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024}
    }
```
