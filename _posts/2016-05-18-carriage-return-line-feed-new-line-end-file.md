---
layout: head
layout: post
title: 贝叶斯优化（Bayesian Optimization）
description: 回车和换行在不同系统下面定义不同，时不时会有一些小问题出来，git 经常出现的 No new line at the end of file 也让很多人费解，需要梳理一下
category: blog
---

## 目标问题

贝叶斯优化（Bayesian Optimization）用于解决黑箱无导数全局优化问题（black-box derivative-free global optimization）：

$$
\max_{\mathbf{x}\in\mathcal{X}}\;f(\mathbf{x})\tag{1}
$$

而与传统的优化问题不同，式（1）具有以下特征：

- 目标函数 $f(\cdot)$ 没有闭式表达（black-box），且导数未知或难以计算（derivative-free），仅可以获得在任意采样点 $\mathbf{x}$ 处的取值 $y=f(\mathbf{x})$ ;
- 对目标函数 $f(\cdot)$ 进行采样/观测（Evaluate）的成本很高，例如，$y=f(\mathbf{x})$ 需要耗费大量资源进行仿真实验获得；
- 不假设函数 $f(\cdot)$ 具有特殊性质，其可能非线性、非凸，可能具有多个局部最优点；
- 优化变量 $\mathbf{x}$ 的维度较低（一般 $d\leq20$ ），约束集 $\mathcal{X}$ 为简单集（超箱、单纯性等）。

由于可解决具有上述性质的优化问题，贝叶斯优化在机器学习的超参数优化任务中应用颇多，尤其是对于深度神经网络和强化学习。

## 算法思想

由于没有导数信息，黑箱函数的优化需要使用搜索的方法；而如前文所言，对目标函数 $f(\cdot)$ 进行采样的成本高昂。算法的设计目标即是通过尽可能少的采样次数，获得问题（1）的（全局）最优解。在这个意义上，所有求解黑箱函数优化的算法都可以归类为基于模型的序贯优化（sequential optimization），即每步迭代，都使用一个代理函数/模型去近似目标函数 $f(\cdot)$ ，并根据其决定下一步的最优采样位置。

区别于其他方法，贝叶斯优化使用一个统计模型作为函数 $f(\cdot)$ 的代理（surrogate），且在选择新的采样点时使用了贝叶斯方法。具体而言，贝叶斯优化认为目标函数 $f(\mathbf{x})$ 是某个随机过程的一次实现（先验分布 $P(f)$ ），如下图所示

<center><p><font size="3"><em>随机过程可视为函数的分布，对于任意的自变量 x（指标），其返回函数 f(x) 取值的一个分布（图中虚线）。或者，可以认为，其返回 f(x) 取值的一个估计值（均值 &mu;(x)，图中实线），以及此估计的置信程度（方差 &sigma;(x)，图中紫色区域）</em></font><br/></p></center>

设经过前 $t$ 步迭代，已获得样本集 $\mathcal{D}_ {1:t}$ ；根据贝叶斯理论，我们可以获得目标函数 $f(\mathbf{x})$ 的后验分布：$P(f\vert\mathcal{D}_ {1:t})\propto P(\mathcal{D}_ {1:t}\vert f)P(f)$ 。进而，第 $t+1$ 个采样点可通过最大化某个期望效用函数 $S(\mathbf{x}\vert P(f\vert\mathcal{D}_ {1:t}))$ 进行选取（例如，最大化后验均值 $\mu(\mathbf{x}\vert P(f\vert\mathcal{D}_ {1:t}))$ ），即有

$$
\mathbf{x}_{t+1}\leftarrow\arg\,\max_{\mathbf{x}}\,S(\mathbf{x}\vert P(f\vert\mathcal{D}_{1:t}))\tag{2}
$$

根据（2）式，在贝叶斯优化中，一般称 $S(\mathbf{x}\vert P(f\vert\mathcal{D}_ {1:t}))$ 为获取函数（acquisition function）。在获得 $\mathbf{x}_ {t+1}$ 处的观测值 $y_{t+1}=f(\mathbf{x}_ {t+1})$ 之后。重复上述过程，直至达到采样次数上限 $T$ . 最终，算法返回所有观测值中最大的样本点 $f(\mathbf{x}^*)=y_T^+=\max\{y_1,\cdots,y_t,\cdots,y_T\}$ 作为优化问题（1）的解。

显然，贝叶斯优化可以视为一个序贯优化方法，其每步迭代，都求解原始优化问题的一个近似/代理问题（即 $\max_{\mathbf{x}}\,S(\mathbf{x}\vert P(f\vert\mathcal{D}_ {1:t}))$ ），最终得到原问题的解。而使用统计模型对函数 $f(\cdot)$ 进行建模，其意义主要有两点：

1. 每步迭代，最大化函数 $f(\cdot)$ 取值分布的期望效用，仅追求 “平均” 意义上的最优，（通过设计合适的获取函数）可以较好的平衡 “全局搜索” 和 “局部最优” ，进而使用尽可能少的采样，获得尽可能好的解；
2. 由于使用了概率模型，对于存在观测噪声的情况，即 $y_t=f(\mathbf{x}_ t)+\epsilon_t$ ，可以很容易地将其包括进来。

## \ No new line at end of file

基于上面的原因，再去看 git diff 的`\ No new line at end of file`信息，就很好解释了。

各编辑器对于换行符的理解偏差，导致的文件确实发生了变化，多了或少了最后的`0a`，那么对于 diff 程序来说，这当然是不可忽略的，但因为`0a`是不可见字符，并且是长久以来的历史原因，所以 diff 程序有个专门的标记来说明这个变化，就是：

`\ No new line at end of file`

各编辑器也有相应的办法去解决这个问题，比如 Sublime，在`Default/Preferences.sublime-settings`中设置：

    // Set to true to ensure the last line of the file ends in a newline
    // character when saving
    "ensure_newline_at_eof_on_save": true,

所以，请遵守规范。

## 测试段落

这是一个测试段落，by Jhon Hu.

行内公式测试一：$\mathbf{A},\,\mathbb{A},\,\mathcal{A}$ .

行内公式测试二：$\mathbf{A}=(\mathbf{B}+\lambda\mathbf{I})^{-1}$ .

行间公式测试：

$$
(\mathbf{A}+\mathbf{x}\mathbf{y}^\mathrm{T})^{-1}=\mathbf{A}^{-1}-\frac{\mathbf{A}^{-1}\mathbf{x}\mathbf{y}^\mathrm{T}\mathbf{A}^{-1}}{1+\mathbf{y}^\mathrm{T}\mathbf{A}^{-1}\mathbf{x}}
$$

测试完毕。

[BeiYuu]:    http://beiyuu.com  "BeiYuu"
