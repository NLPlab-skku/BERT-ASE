# BERT-ASE
This repository contains the training code and data for the paper
[Never Too Late to Learn: Regularizing Gender Bias in Coreference Resolution], published in WSDM 2023.

## Overview
BERT-ASE, alleviates the gender bias problems in the public released BERT by two bias mitigation methods.
Our bias mitigation methods, Stereotype Neutralization (SN) and EWC, enable the PLMs to find proper gender pronouns in the given context without sterotypical or skewed misconceptions.

## Reproducing Experiments
Before you start, you need to download the WinoBias datasets available on the [corefBias] (https://github.com/uclanlp/corefBias).
Our preprocessing codes referenced previous work by By Daniel de Vassimon Manela, Boris van Breugel, Tom Fisher, David Errington. (https://github.com/12kleingordon34/NLP_masters_project)

### Training

```shell
python finetune_both.py \
  --do_train
  --data augmented
```
