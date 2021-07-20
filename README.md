# decoding-biases

## Overview

This repo contains code used in the decoding experiments from various NLG fairness metrics [this paper](https://arxiv.org/abs/2105.04054), which can be cited as follows:

```
@inproceedings{sheng2021societal,
  title={Societal Biases in Language Generation: Progress and Challenges},
  author={Sheng, Emily and Chang, Kai-Wei and Natarajan, Premkumar and Peng, Nanyun},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing},
  year={2021}
}
```

1. The _regard_ metric is from: [The Woman Worked as a Babysitter: On Biases in Language Generation](https://arxiv.org/abs/1909.01326). The code + classifier can be found [here](https://github.com/ewsheng/nlg-bias).

2. The African American English/White-Aligned English evaluations are from [Investigating African-American Vernacular English in Transformer-Based Text Generation](https://arxiv.org/abs/2010.02510), and the dataset can be found [here](https://github.com/sophiegroenwold/AAVE_SAE_dataset).

3. The individual/group fairness distributional metrics are from [Reducing Sentiment Bias in Language Models via Counterfactual Evaluation](https://arxiv.org/abs/1911.03064). 

4. The gendered word co-occurrence score metric is from [Identifying and Reducing Gender Bias in Word-Level Language Models](https://arxiv.org/abs/1904.03035).

5. `data/female_word_list.txt` and `data/male_word_list.txt` are taken from [here](https://github.com/uclanlp/gn_glove/tree/master/wordlist).

## Running Scripts
To run scripts, first run:
```
conda create --name decoding-biases python==3.7
conda activate decoding-biases
pip install -r requirements.txt
```

### Sample Generation

To generate samples, you can run:
```
python generate.py \
--evaluation regard \
--model_type gpt2 \
--decode_type greedy
```
Run ```python generate.py -h``` to see all options.

To run the `aae-wae` generation/evaluation, you'll have to contact the authors of the dataset [here](https://github.com/sophiegroenwold/AAVE_SAE_dataset) to obtain the prompts and then put the `aae_samples.tsv` and `wae_samples.tsv` samples in `data/`. 

The current script will generate 100 samples per prompt if the evaluation is `regard` and 1 sample per prompt for all other evaluations, consistent with what is described in the original paper.

### Evaluation

#### _Regard_
To run the _regard_ evaluations on the generated samples, you'll have to first download the _regard_ classifier [here](https://github.com/ewsheng/nlg-bias).
Since the classifier was trained with demographics masked out with "XYZ", we suggest doing the same with the generated samples.
In other words, you can take the file of samples generated with `generate.py` (e.g., `gpt2.greedy.regard.csv`), replace demographics with `XYZ`, input the resulting file to the _regard_ classifier, and use the file output by the classifier as the `regard_file` below.

To then run the regard evaluation:
```
python evaluate.py \
--evaluation regard \
--model_type gpt2 \
--decode_type greedy \
--regard_file [prediction file from regard classifier] \
--unmasked_regard_file gpt2.greedy.regard.csv
```

#### AAE-WAE 
To run the aae-wae evaluation:
```
python evaluate.py \
--evaluation aae-wae \
--model_type gpt2 \
--decode_type greedy \
--aae_wae_sentiment_file gpt2.greedy.aae-wae.csv
```

#### Individual/Group Distributional Fairness
To run the IF/GF evaluation:
```
python evaluate.py \
--evaluation distrib \
--model_type gpt2 \
--decode_type greedy \
--distrib_file gpt2.greedy.distrib.csv
```

#### Gendered Word Co-occurrence
To run the gendered word co-occurrence score evaluation as described in the original paper, you'll have to have generated samples for the other evaluation: `regard`, `distrib`, and `aae-wae`. Then, run the following:
```
python evaluate.py \
--evaluation ratio \
--model_type gpt2 \
--decode_type greedy \
--regard_file [prediction file from regard classifier] \
--unmasked_regard_file gpt2.greedy.regard.csv \
--distrib_file gpt2.greedy.distrib.csv \
--aae_wae_sentiment_file gpt2.greedy.aae-wae.csv
```
