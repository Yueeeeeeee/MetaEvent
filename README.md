# Introduction

The MetaEvent repository is the PyTorch implementation of ACL 2023 Paper [Zero-and Few-Shot Event Detection via Prompt-Based Meta Learning](https://arxiv.org/abs/2305.17373)

<img src=pics/intro.png>

We propose MetaEvent, a meta learning-based framework for zero- and few-shot event detection. Specifically, we sample training tasks from existing event types and perform meta training to search for optimal parameters that quickly adapt to unseen tasks. In our framework, we propose to use the cloze-based prompt and a trigger-aware soft verbalizer to efficiently project output to unseen event types. Moreover, we design a contrastive meta objective based on maximum mean discrepancy (MMD) to learn class-separating features. As such, the proposed MetaEvent can perform zero-shot event detection by mapping features to event types without any prior knowledge. In our experiments, we demonstrate the effectiveness of MetaEvent in both zero-shot and few-shot scenarios, where the proposed method achieves state-of-the-art performance in extensive experiments on benchmark datasets FewEvent and MAVEN.


## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{yue2023zero,
  title={Zero-and Few-Shot Event Detection via Prompt-Based Meta Learning},
  author={Yue, Zhenrui and Zeng, Huimin and Lan, Mengfei and Ji, Heng and Wang, Dong},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```


## Data & Requirements

The adopted datasets are publicly available, you can download FewEvent [here](https://github.com/congxin95/PA-CRF/tree/main/data/FewEvent) and MAVEN [here](https://github.com/chen700564/causalFSED/tree/master/data/maven). To run our code you need PyTorch & Transformers, see requirements.txt for our running environment


## Run MetaAdapt

```bash
python src/train_fewevent_zeroshot.py;
```
Excecute the above command (with arguments) to perform zero-shot training on FewEvent.
```bash
python src/train_maven_zeroshot.py;
```
Excecute the above command (with arguments) to perform zero-shot training on MAVEN.
```bash
python src/train_fewevent_fewshot.py;
```
Excecute the above command (with arguments) to perform few-shot training on FewEvent.
```bash
python src/train_maven_fewshot.py;
```
Excecute the above command (with arguments) to perform few-shot training on MAVEN.


## Performance

The zero and few-shot performance on both datasets is presented below. For training and evaluation details, please refer to our paper and code.

<img src=pics/zeroshot_performance.png width=1000>
<img src=pics/fewshot_performance.png width=1000>


## Acknowledgement

During the implementation we base our code mostly on [PA-CRF](https://github.com/congxin95/PA-CRF) from Cong et al. and [MetaST](https://github.com/microsoft/MetaST) by Wang et al. Many thanks to these authors for their great work!