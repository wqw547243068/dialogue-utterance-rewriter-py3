# dialogue-utterance-rewriter-py3

## 多轮对话改写
- [ACL 2019 | 使用表达改写提升多轮对话系统效果](https://www.aminer.cn/research_report/5d527dd4d5e908133c946b07)
![](http://zhengwen.aminer.cn/a1.png)
- 模型框架
![](http://zhengwen.aminer.cn/a2.png)
- 对于任务导向型对话系统和闲聊型对话系统均有效果提升，实现了用更成熟的单轮对话技术解决多轮对话问题。
![](http://zhengwen.aminer.cn/a10.png)
- [Transformer多轮对话改写实践](https://zhuanlan.zhihu.com/p/137127209)
   - 介绍了多轮对话存在指代和信息省略的问题，同时提出了一种新方法-抽取式多轮对话改写，可以更加实用的部署于线上对话系统，并且提升对话效果
- 日常的交流对话中
   - 30%的对话会包含指代词。比如“它”用来指代物，“那边”用来指代地址
   - 同时有50%以上的对话会有信息省略
- 对话改写的任务有两个：1这句话要不要改写；2 把信息省略和指代识别出来。对于baseline论文放出的数据集，有90%的数据都是简单改写，也就是满足任务2，只有信息省略或者指代词。少数改写语句比较复杂，本文训练集剔除他们，但是验证集保留。
![](https://pic1.zhimg.com/80/v2-d80efd57b81c6ece955a247ca7247db4_1440w.jpg)
- Transformers结构可以通过attention机制有效提取指代词和上下文中关键信息的配对，最近也有一篇很好的工作专门用Bert来做指代消岐[2]。经过transformer结构提取文本特征后，模型结构及输出如下图。

![](https://pic4.zhimg.com/80/v2-0c4a789b68c60c8279dbd98fc18b5b2b_1440w.jpg)
- 输出五个指针中：
   - 关键信息的start和end专门用来识别需要为下文做信息补全或者指代的词；
   - 补全位置用来预测关键信息(start-end)插入在待改写语句的位置，实验中用插入位置的下一个token来表示；
   - 指代start和end用来识别带改写语句出现的指代词。
   - 当待改写语句中不存在指代词或者关键信息的补全时，指代的start和end将会指向cls，同理补全位置也这样。如同阅读理解任务中不存在答案一样，这样的操作在做预测任务时，当指代和补全位置的预测最大概率都位于cls时就可以避免改写，从而保证了改写的稳定性。
- 效果评估
   - 准确度
   ![](https://pic3.zhimg.com/80/v2-75faf2bed618cf5170efa56d65cd88e2_1440w.jpg)
   - 性能
   ![](https://pic3.zhimg.com/80/v2-75faf2bed618cf5170efa56d65cd88e2_1440w.jpg)
   - 对数据集的依赖
   ![](https://pic2.zhimg.com/80/v2-b22df166f10a6716b3db01e303f1a721_1440w.jpg)
   - 对负样本(不需要改写样本)的识别.基于指针抽取的方法对负样本的识别效果会更好。同时根据对长文本的改写效果观察，生成式改写效果较差。
   ![](https://pic3.zhimg.com/80/v2-b653c7da5923236991b6b0c5f973703a_1440w.jpg)
- 示例
   - 从左到右依次是A1|B1|A2|算法改写结果|用户标注label
>- 你知道板泉井水吗 | 知道 | 她是歌手 | 板泉井水是歌手 | 板泉井水是歌手
>- 乌龙茶 | 乌龙茶好喝吗 | 嗯好喝 | 嗯乌龙茶好喝 | 嗯乌龙茶好喝
>- 武林外传 | 超爱武林外传的 | 它的导演是谁 | 武林外传的导演是谁 | 武林外传的导演是谁
>- 李文雯你爱我吗 | 李文雯是哪位啊 | 她是我女朋友 | 李文雯是我女朋友 | 李文雯是我女朋友
>- 舒马赫 | 舒马赫看球了么 | 看了 | 舒马赫看了 | 舒马赫看球了

- 结论
   - 抽取式文本改写和生成式改写效果相当
   - 抽取式文本改写速度上绝对优于生成式
   - 抽取式文本改写对训练数据依赖少
   - 抽取式文本改写对负样本识别准确率略高于生成式




## installation

This is an unoffical repo adapted from [Repo](https://github.com/chin-gyou/dialogue-utterance-rewriter). Followings are the differences:

| Desc | Official | This |
|:---:|:---:|:---:|
| Python Version | Python 2.7 | Python 2.7 - 3.7 |
| Tensorflow Version | 1.4 | 1.14 |
| CUDA Version | CUDA8.0 | CUDA10.0 |
| Code Run | Some issues | Can run |
| Platform | Linux | Support both Linux & Windows |


## dialogue-utterance-rewriter-corpus

Dataset for ACL 2019 paper "[Improving Multi-turn Dialogue Modelling with Utterance ReWriter
](https://arxiv.org/abs/1906.07004)"

After another two months of human labeling, we release a much more better quality dataset(only positive samples) than the original one we used in our paper for better research.  Hope you can get a better result. 

### Description

The positive dataset, 20000 dialogs. Each line in corpus.txt consists of four utterances of dialog (two context utterances, current utterance), and the rewritten uterance. Each line is `tab-delimited` (one tab) with the following format:

```bash
<A: context_1>\t<B: context_2>\t<A: current>\t<A: A: rewritten current>
```

## LSTM-based Model
### About the code
This code is based on the [Pointer-Generator](https://github.com/abisee/pointer-generator) code. 

**Requirements**

To run the souce codes, some external packages are required

* python 2.7
* Tensorflow  1.4

vocab file:
```bash
<word>\t<count>
```
### Run training and Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:
To train your model, run:

```
sh train.sh
sh val.sh
```
### Run beam search decoding
To run beam search decoding, first set restore_best_model=1 to restore the best model.

```
sh train.sh
sh test.sh
```
**Why can't you release the Transformer model?** Due to the company legal policy reasons, we cannot realease the Transformer code which has been used in online environment. However, feel free to email us to discuss training and model details. 

### Citation

```
@article{su2019improving,
  title={Improving Multi-turn Dialogue Modelling with Utterance ReWriter},
  author={Su, Hui and Shen, Xiaoyu and Zhang, Rongzhi and Sun, Fei and Hu, Pengwei and Niu, Cheng and Zhou, Jie},
  journal={ACL},
  year={2019}
}
```


