# BERT-ASE
This repository contains the training code and data for the paper
[Never Too Late to Learn: Regularizing Gender Bias in Coreference Resolution], published in WSDM 2023. [pdf](https://dl.acm.org/doi/pdf/10.1145/3539597.3570473)

## ☀ Overview
BERT-ASE, alleviates the gender bias problems in the public released BERT by two bias mitigation methods. </br>
Our bias mitigation methods, Stereotype Neutralization (SN) and EWC, enable the PLMs to find proper gender pronouns in the given context without sterotypical or skewed misconceptions.

## 📖 Reproducing Experiments
Before you start, you need to download the WinoBias datasets available on the [corefBias](https://github.com/uclanlp/corefBias). </br>
Our preprocessing codes referenced previous work by By Daniel de Vassimon Manela, Boris van Breugel, Tom Fisher, David Errington. [github link](https://github.com/12kleingordon34/NLP_masters_project)

### 🔥 Training

```shell
python finetune_both.py \
  --do_train
  --data augmented
```
