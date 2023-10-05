# KoSimCSE Training on Amazon SageMaker

## Introduction

[SimCSE](https://aclanthology.org/2021.emnlp-main.552/) is a highly efficient and innovative embedding technique based on the concept of contrastive learning. Unsupervised learning can be performed without the need to prepare ground-truth labels, and high-performance supervised learning can be performed if a good NLI (Natural Language Inference) dataset is prepared. The concept is very simple and the psudeo-code is intuitive, so the implementation is not difficult, but I have seen many people still struggle to train this model.

The official implementation code from the authors of the paper is publicly available, but it is not suitable for a step-by-step implementation. Therefore, we have reorganized the code based on [Simple-SIMCSE's GitHub](https://github.com/hppRC/simple-simcse) so that even ML beginners can train the model from the scratch with a step-by-step implementation. It's minimalist code for beginners, but data scientists and ML engineers can also make good use of it.

### Added over Simple-SimCSE
- Added the Supervised Learning part, which shows you step-by-step how to construct the training dataset.
- Added Distributed Learning Logic. If you have a multi-GPU setup, you can train faster.
- Added SageMaker Training. `ml.g4dn.xlarge` trains well, but we recommend `ml.g4dn.12xlarge` or` ml.g5.12xlarge` for faster training.

## Requirements

We recommend preparing an Amazon SageMaker instance with the specifications below to perform this hands-on.

### SageMaker Notebook instance
- `ml.g4dn.xlarge`

### SageMaker Training instance
- `ml.g4dn.xlarge` (Minimum)
- `ml.g5.12xlarge` (Recommended)

## Datasets

For supervised learning, you need an NLI dataset that specifies the relationship between the two sentences. For unsupervised learning, we recommend using wikipedia raw data separated into sentences. This hands-on uses the dataset registered with huggingface, but you can also configure your own dataset.

The datasets used in this hands-on are as follows

#### Supervised
- [Klue-NLI](https://huggingface.co/datasets/klue/viewer/nli/)
- [Kor-NLI](https://huggingface.co/datasets/kor_nli)

#### Unsupervised 
- [kowiki-sentences](https://huggingface.co/datasets/heegyu/kowiki-sentences): Data from 20221001 Korean wiki split into sentences using kss (backend=mecab) morphological analyzer.


## How to run

### JupyterLab / Jupyter Notebook / Colab

#### Supervised
- `A.1_sup-prepare-nli-dataset.ipynb`: Prepare Dataset for training
- `A.2_sup-train-dev.ipynb`: Training on Local Environment
- `A.3_sm_sup_train.ipynb`: Training on SageMaker

#### Unsupervised 
- `B.2_unsup-train-dev.ipynb`: Training on Local Environment
- `B.3_sm_unsup_train.ipynb`: Training on SageMaker

### Command line (Local)

#### Supervised
```bash
cd src
python sup_prepare_dataset.py
bash sup_run_local.sh
```

#### Unsupervised 
```bash
cd src
bash unsup_run_local.sh
```

## Performance 

### KLUE-STS
| Model                  | AVG | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSimCSE-RoBERTa-base (Unsupervised) | 81.17 | 81.27 | 80.96 | 81.70 | 80.97 | 81.63 | 80.89 | 81.12 | 80.81 |
| KoSimCSE-RoBERTa-base (Supervised) | 84.19 | 83.04 | 84.46 | 84.97 | 84.50 | 84.95 | 84.45 | 82.88 | 84.28 |

### Kor-STS
| Model                  | AVG | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSimCSE-RoBERTa-base (Unsupervised) | 81.20 | 81.53 | 81.17 | 80.89 | 81.20 | 80.93 | 81.22 | 81.48 | 81.14 |
| KoSimCSE-RoBERTa-base (Supervised) | 85.33 | 85.16 | 85.46 | 85.37 | 85.45 | 85.31 | 85.37 | 85.13 | 85.41 |

## References
- Simple-SimCSE: https://github.com/hppRC/simple-simcse
- KoSimCSE: https://github.com/BM-K/KoSimCSE-SKT
- SimCSE (official): https://github.com/princeton-nlp/SimCSE
- SimCSE paper: https://aclanthology.org/2021.emnlp-main.552

## License Summary

This sample code is provided under the MIT-0 license. See the license file.