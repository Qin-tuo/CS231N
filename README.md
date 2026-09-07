# CS231n: Deep Learning for Computer Vision — Spring 2025

Stanford CS231n 2025 春季课程资料存档（由李飞飞 Fei-Fei Li 等主讲）。
官方主页存档：<https://cs231n.stanford.edu/2025/>

本目录包含：**Slides 幻灯片**、**Assignments 作业**、**Reading materials 阅读资料（课程笔记 / 讲义 / 论文）**。

> 注：**课程视频**托管在 Stanford Canvas（需 Stanford 账号登录），无法匿名下载，故未包含。
> 链接：<https://canvas.stanford.edu/courses/205350/external_tools/69960>

---

## 目录结构

```
cs231n-2025/
├── slides/            幻灯片
│   ├── 2025/          第1–17讲 + 讨论课 section_2/3/5/6
│   ├── 2024/          第18讲、第6讲备用版(part1/part2/review)
│   └── 2021/          TensorFlow 讨论课 notebook
├── assignments/       三次作业(starter code zip + 说明 md + 解压目录)
│   ├── assignment1_colab.zip  + assignment1/
│   ├── assignment2_colab.zip  + assignment2/
│   └── assignment3_colab.zip  + assignment3/
└── readings/          阅读资料
    ├── notes/         官方课程笔记(cs231n.github.io, markdown源)
    ├── handouts/      数学讲义(求导 / 线性反向传播)
    └── papers/        课程指定阅读的经典论文(PDF)
```

---

## 课程安排 · 逐讲索引（讲题 / 幻灯片 / 阅读）

| # | 日期 | 讲题 | 主讲 | 幻灯片 | 相关阅读 |
|---|------|------|------|--------|----------|
| 1 | 04/01 | Introduction（计算机视觉与课程概览） | Fei-Fei Li, Ehsan Adeli | `slides/2025/lecture_1_part_1.pdf`, `lecture_1_part_2.pdf` | notes: python-numpy-tutorial |
| 2 | 04/03 | Image Classification with Linear Classifiers | — | `slides/2025/lecture_2.pdf` | notes: classification, linear-classify |
| 3 | 04/08 | Regularization and Optimization | Zane Durante | `slides/2025/lecture_3.pdf` | notes: optimization-1；博客: Why Momentum Really Works |
| 4 | 04/10 | Neural Networks and Backpropagation | Ehsan Adeli | `slides/2025/lecture_4.pdf` | handouts: derivatives, linear-backprop；notes: optimization-2；博客: colah Backprop, NN&DL ch.2 |
| 5 | 04/15 | Image Classification with CNNs | Justin Johnson | `slides/2025/lecture_5.pdf` | notes: convolutional-networks |
| 6 | 04/17 | CNN Architectures（BN / 迁移学习 / AlexNet,VGG,ResNet） | Zane Durante | `slides/2025/lecture_6.pdf`（备用: `slides/2024/lecture_6_part_1/2/review`） | papers: LeNet, AlexNet, VGG, GoogLeNet, ResNet |
| 7 | 04/22 | Recurrent Neural Networks（RNN/LSTM/GRU） | Zane Durante | `slides/2025/lecture_7.pdf` | 博客: DL book RNN 章、colah Understanding LSTMs |
| 8 | 04/24 | Attention and Transformers | Justin Johnson | `slides/2025/lecture_8.pdf` | papers: Transformer；博客: Illustrated Transformer, Lilian Weng Attention |
| 9 | 04/29 | Object Detection, Segmentation, Visualizing & Understanding | — | `slides/2025/lecture_9.pdf` | papers: R-CNN, Fast/Faster R-CNN, YOLO, FCN, DETR |
| 10 | 05/01 | Video Understanding | Ruohan Gao | `slides/2025/lecture_10.pdf` | — |
| 11 | 05/06 | Large Scale Distributed Training | Justin Johnson | `slides/2025/lecture_11.pdf` | — |
| 12 | 05/08 | Self-supervised Learning | Ehsan Adeli | `slides/2025/lecture_12.pdf` | papers: ViT, DINO；博客: Lilian Weng Self-Supervised |
| 13 | 05/15 | Generative Models 1（VAE/GAN/自回归） | Justin Johnson | `slides/2025/lecture_13.pdf` | 博客: ELBO — What & Why |
| 14 | 05/20 | Generative Models 2（Diffusion 扩散模型） | Justin Johnson | `slides/2025/lecture_14.pdf` | — |
| 15 | 05/22 | 3D Vision | Jiajun Wu | `slides/2025/lecture_15.pdf` | — |
| 16 | 05/27 | Vision and Language | Ranjay Krishna | `slides/2025/lecture_16.pdf` | — |
| 17 | 05/29 | Robot Learning（深度强化学习 / 机器人操作） | Yunzhu Li | `slides/2025/lecture_17.pdf` | — |
| 18 | 06/03 | Human-Centered AI | Fei-Fei Li | `slides/2024/lecture_18.pdf` | — |

讨论课（sections）：`slides/2025/section_2.pdf`、`section_3.pdf`、`section_5.pdf`、`section_6.pdf`；
早期 TensorFlow 讨论课：`slides/2021/discussion_4_tensorflow.ipynb`。

---

## 作业 Assignments

| 作业 | 占比 | 内容 |
|------|------|------|
| Assignment 1 | 12% | 图像分类、kNN、Softmax、全连接神经网络 |
| Assignment 2 | 18% | Batch Normalization、Dropout、卷积网络、网络可视化、RNN 图像描述 |
| Assignment 3 | 15% | Transformer 图像描述、自监督学习、扩散模型、CLIP 与 DINO |

每个作业含 starter code（`assignmentN_colab.zip`，已解压到 `assignmentN/`）与说明 `assignmentN.md`。
官方作业页：<https://cs231n.github.io/assignments2025/>（提交走 Gradescope）。

---

## 阅读资料 Readings

- **课程笔记 `readings/notes/`**（官方 markdown 源，含公式 LaTeX）：
  image classification, linear classification, optimization-1/2, convolutional networks,
  neural networks 1/2/3 + case study, python/numpy 教程, jupyter/colab 教程。
  在线阅读：<https://cs231n.github.io/>
- **数学讲义 `readings/handouts/`**：`derivatives.pdf`（向量/矩阵求导）、`linear-backprop.pdf`（线性层反向传播推导）。
- **经典论文 `readings/papers/`**：LeNet(1998)、AlexNet(2012)、VGG、GoogLeNet、FCN、R-CNN/Fast/Faster R-CNN、YOLO、ResNet、Transformer、DETR、ViT、DINO。

### 仅链接（外部博客/文章，未下载）

- Why Momentum Really Works — <https://distill.pub/2017/momentum/>
- Calculus on Computational Graphs: Backprop — <https://colah.github.io/posts/2015-08-Backprop/>
- Neural Networks and Deep Learning, ch.2 — <http://neuralnetworksanddeeplearning.com/chap2.html>
- Understanding LSTM Networks — <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
- The Illustrated Transformer — <https://jalammar.github.io/illustrated-transformer/>
- Attention? Attention! (Lilian Weng) — <https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html>
- Self-Supervised Learning (Lilian Weng) — <https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html>
- ELBO — What & Why — <https://yunfanj.com/blog/2021/01/11/ELBO.html>
- Deep Learning Book, RNN chapter — <http://www.deeplearningbook.org/contents/rnn.html>

---

*资料来源：cs231n.stanford.edu 与 cs231n.github.io（Stanford CS231n Spring 2025）。仅供个人学习使用，版权归 Stanford 及各作者所有。*
