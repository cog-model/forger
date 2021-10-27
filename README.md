# ForgER: Forgetful Expirience Replay for Reinforcement Learning from Demonstrations


This repository is the implementation version using Newest MineRL dataset. The original repo and info please see below. 


Dependencies:

##### Python == 3.7 
##### tensorflow == 2.5.1
##### tensorflow-estimator == 2.3.0
##### tensorflow-probability == 0.18.0
##### gym == 0.19.0
##### chainerrl == 0.8.0
##### tqdm
##### cpprb == 9.3.2
##### opencv-python == 4.4.0.46
##### pyyaml == 5.4
##### wandb 
##### minerl == 0.3.7 

<br>

Please download MineRL dataset through:

##### import minerl
##### minerl.data.download('demonstrations/') 

<br><br>
====================================== Original Version ======================================

This repository is the TF2.0 implementation of Forgetful Replay Buffer for Reinforcement Learning from Demonstrations by [Alexey Skrynnik](https://github.com/Tviskaron), [Aleksey Staroverov](https://github.com/alstar8), [Ermek Aitygulov](https://github.com/ermekaitygulov), [Kirill Aksenov](https://github.com/axellkir), [Vasilii Davydov](https://github.com/dexfrost89), [Aleksandr I. Panov](https://github.com/grafft). 

[[Paper]](https://arxiv.org/abs/2006.09939) [[Webpage]](https://sites.google.com/view/forgetful-experience-replay)

![Forging phase](static/forging.png)

## Citation
If you use this repo in your research, please consider citing the paper as follows
```
@misc{skrynnik2020forgetful,
    title={Forgetful Experience Replay in Hierarchical Reinforcement Learning from Demonstrations},
    author={Alexey Skrynnik and Aleksey Staroverov and Ermek Aitygulov and Kirill Aksenov and Vasilii Davydov and Aleksandr I. Panov},
    year={2020},
    eprint={2006.09939},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Demonstrations 
For a set of simple environments, most of the expert data is already in the demonstrations folder.
To train ForgER on *MineRL* domain you need to put expert data in the ```demonstrations``` folder.

## Training

To train ForgER on *Simple set*, run this command:

```train
python train_simple_set.py --config simple_set_config.yaml
```
You can change the environment and the path to expert data in the config. 
The ```simple_set_config.yaml``` file provides an example config for the Acrobot-v1 environment. 



To train ForgER on *Treechop*, run this command:

```train
python train_treechop.py --config treechop_config.yaml
```

To train ForgER on *MineRL*, run this command:

```train
python train_minerl.py --config minerl_config.yaml
```

## Results on MineRLObtainDiamond-v0 (1000 seeds)

| Item | MineRL2019 | ForgER | ForgER++|
| --- | --- | --- | --- |
| log | 859 | **882** | 867 |
| planks | 805 | **806** | 792 |
| stick | 718 | 747 | **790** |
| crafting table | 716 | 744 | **790** |
| wooden pickaxe | 713 | 744 | **789** |
| cobblestone | 687 | 730 | **779** |
| stone pickaxe | 642 | 698 | **751** |
| furnace | 19 | 48 | **98** |
| iron ore | 96 | 109 | **231** |
| iron ingot | 19 | 48 | **98** |
| iron pickaxe | 12 | 43 | **83** |
| diamond | 0 | 0 | **1** |
| mean reward | 57.701 | 74.09 | **104.315** |
