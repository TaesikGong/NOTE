# 📝NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation (NeurIPS '22)

This is the official PyTorch Implementation of "NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation (NeurIPS '22)" by [Taesik Gong](https://taesikgong.com/), [Jongheon Jeong](https://jh-jeong.github.io/), Taewon Kim, [Yewon Kim](https://yewon-kim.com/), [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html), and [Sung-Ju Lee](https://sites.google.com/site/wewantsj/).

[ [arXiv](https://arxiv.org/abs/2208.05117) ] [ [Website](https://nmsl.kaist.ac.kr/projects/note/) ]

## Installation Guide

1. Download or clone our repository
2. Set up a python environment using conda (see below)
3. Prepare datasets (see below)
4. Run the code (see below)

## Python Environment

We use [Conda environment](https://docs.conda.io/).
You can get conda by installing [Anaconda](https://www.anaconda.com/) first.

We share our python environment that contains all required python packages. Please refer to the `./note.yml` file

You can import our environment using conda:

    conda env create -f note.yml -n note

Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Prepare Datasets

This code reproduces the results in Table 1:

- CIFAR10-C
- CIFAR100-C
- ImageNet-C

To run our codes, you first need to download at least one of the datasets. Run the following commands:

    $cd .                           # project root
    $. download_cifar10c.sh        # download CIFAR10/CIFAR10-C datasets
    $. download_cifar100c.sh       # download CIFAR100/CIFAR100-C datasets
    $cat download_imagenetc.md       # instruction for ImageNet/ImageNet-C datasets (please refer to the instruction)


## Run

### Prepare Source model

"Source model" refers to a model that is trained with the source (clean) data only. Source models are required to all methods to perform test-time adaptation. 

We provide pre-trained models:
- [CIFAR10-C](https://drive.google.com/file/d/1YsyHY3rFCaWWDTOVh-RuI1I2bJ5i9Yey/view?usp=sharing)
- [CIFAR100-C](https://drive.google.com/file/d/10JVqrgnTlG9Mbj7ogNww2MZZHHlFyzB8/view?usp=sharing)
- [ImageNet-C](https://drive.google.com/file/d/18moC1ji6B_5NMF3VO7NW50hb-R4G_Ns2/view?usp=sharing)

After extracting `log.zip`, put this folder to the project root directory, e.g., `NOTE/log`.
For CIFAR10-C and CIFAR100-C, pre-trained models consist of without-IABN layers and with-IABN layers under three random seeds (0,1,2). For ImageNet-C, the pre-trained model is with-IABN layers under one random seed (0) because we use the torchvision-provided ResNet model for the without-IABN case.


Alternatively, you can train source models via:

    $. train_src.sh                 #generate source models for CIFAR10 as default.

You can specify which dataset to use in the script file.

### Run Test-Time Adaptation (TTA)

Given source models are available, you can run TTA via:

    $. tta.sh                       #Run NOTE for CIFAR10 as default.

You can specify which dataset and which method in the script file.

### Results on CIFAR-10 CIFAR-100, temporally correlated test stream

|          | CIFAR10-C  | CIFAR100-C | Avg  |
| -------- | :--------: | :--------: | :--: |
| Source   | 42.3 ± 1.1 | 66.6 ± 0.1 | 54.4 |
| BN Stats | 73.4 ± 1.3 | 65.0 ± 0.3 | 69.2 |
| ONDA     | 63.6 ± 1.0 | 49.6 ± 0.3 | 56.6 |
| PL       | 75.4 ± 1.8 | 66.4 ± 0.4 | 70.9 |
| TENT     | 76.4 ± 2.7 | 66.9 ± 0.6 | 71.7 |
| LAME     | 36.2 ± 1.3 | 63.3 ± 0.3 | 49.7 |
| CoTTA    | 75.5 ± 0.7 | 64.2 ± 0.2 | 69.8 |
| NOTE     | 21.1 ± 0.6 | 47.0 ± 0.1 | 34.0 |

### Results on CIFAR-10 CIFAR-100, i.i.d. test stream

|          | CIFAR10-C  | CIFAR100-C | Avg  |
| -------- | :--------: | :--------: | :--: |
| Source   | 42.3 ± 1.1 | 66.6 ± 0.1 | 54.4 |
| BN Stats | 21.6 ± 0.4 | 46.6 ± 0.2 | 34.1 |
| ONDA     | 21.7 ± 0.4 | 46.5 ± 0.1 | 34.1 |
| PL       | 21.6 ± 0.2 | 43.1 ± 0.3 | 32.3 |
| TENT     | 18.8 ± 0.2 | 40.3 ± 0.2 | 29.6 |
| LAME     | 44.1 ± 0.5 | 68.8 ± 0.1 | 56.4 |
| CoTTA    | 17.8 ± 0.3 | 44.3 ± 0.2 | 31.1 |
| NOTE     | 20.1 ± 0.5 | 46.4 ± 0.0 | 33.2 |
| NOTE\*   | 17.6 ± 0.3 | 41.0 ± 0.2 | 29.3 |

## Log

### Raw logs

In addition to console outputs, the result will be saved as a log file with the following structure: `./log/{DATASET}/{METHOD}/{TGT}/{LOG_PREFIX}_{SEED}_{DIST}/online_eval.json`

### Obtaining results

In order to print the classification errors(%) on test set, run the following commands:

    $python eval_script.py --dataset cifar10 --method note --seed all    #print the result of the specified condition.
    $python eval_script.py --dataset all --method all --seed all         #print the entire results.

## Tested Environment

We tested our codes under this environment.

- OS: Ubuntu 20.04.4 LTS
- GPU: NVIDIA GeForce RTX 3090
- GPU Driver Version: 470.74
- CUDA Version: 11.4

## Citation

```
@inproceedings{gong2022note,
    author = {Gong, Taesik and Jeong, Jongheon and Kim, Taewon and Kim, Yewon and Shin, Jinwoo and Lee, Sung-Ju},
    title = {{NOTE}: Robust Continual Test-time Adaptation Against Temporal Correlation},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2022}
}
```
