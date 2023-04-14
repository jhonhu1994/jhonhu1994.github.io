---
layout: head
layout: post
title: Learning to Optimize (L2O)
description: 贝叶斯优化可用于解决黑箱无导数全局优化问题，近年来在机器学习的超参数优化任务中应用颇多，尤其对于深度神经网络和强化学习。
category: blog
---

## 目标问题

对于一个连续优化问题 $\min _{\mathbf{x}\in\mathcal{X}}\; f(\mathbf{x})$ ，可以使用迭代的方法获得其（局部）最优解：
$$
\mathbf{x}_ {t+1}\leftarrow\mathbf{x}_ {t}+\mathbf{g}(\mathbf{z}_ t;\phi)\tag{1}
$$
其中，$\mathbf{z}_ t$ 是第 $t$ 步迭代时可用的信息（例如，已有的迭代点 $\mathbf{x}_ 0,\,\mathbf{x}_ 1,\,\cdots,\,\mathbf{x}_ t$ 以及它们的梯度信息等）；$\mathbf{g}$ 为某个映射函数，一般由 $\phi$ 参数化。Learning to Optimize（L2O）的任务即是在映射函数 $\mathbf{g}$ 的参数空间内学习一个（对于某类问题 $f(\mathbf{x};\theta)$）好的 $\phi$。整体而言，现有的 L2O 方法可分为 model-free 和 model-based 两类。前者使用通用的神经网络架构实现 $\mathbf{g}$，不参考任何已知的解析优化算法。后者则是针对一个现有的解析优化方法，将其整体或部分的迭代更新规则建模为一个可学习的架构，是目前的新方向。 

## Model-Free L2O

### 1. LSTM-based L2O

![LSTM-based L2O](C:\Users\Jhon Hu\iCloudDrive\Documents\Blog\jhonhu1994.github.io-main\images\Learningtooptimize\LSTM-based_L2O.png)

<center><p><font size="3"><em>Fig 1. LSTM-based L2O (unrolled form)</em></font><br/></p></center>

优化过程可以看作是迭代更新 $\mathbf{x}$ 的轨迹，故使用循环神经网络（recurrent neural network，RNN）对更新规则建模是自然的选择。图 1 给出了使用一个长短时记忆网络（long short-term memory，LSTM）实现无约束优化问题 $\min _{\mathbf{x}\in\mathbb{R}^n}\, f(\mathbf{x})$ 的一阶更新的框图，具体而言，即有
$$
\mathcal{L}(\phi)=\mathbb{E}_ {f}\left[\sum_ {t=1}^ {T}w_ tf(\mathbf{x}_ t)\right],\;\text{where}\;\;\mathbf{x}_ {t+1}=\mathbf{x}_ t+\mathbf{g}_ t,\;\text{and}\;\begin{bmatrix}\mathbf{g}_ t\\
h_ {t+1}
\end{bmatrix}=m\left(\nabla f(\mathbf{x}_ t),\,h_ t,\,\phi\right),\tag{2}
$$
其中，$m$ 代表 LSTM，其根据当前时刻的输入（当前点的梯度 $\nabla f(\mathbf{x}_ t)$ ）和网络状态 $h_ {t}$ （包含历史梯度信息），输出更新步 $\mathbf{g}_ t$；$\phi$ 是待学习的网络参数，$\mathcal{L}(\cdot)$ 是损失函数（loss function），T 为LSTM 的展开长度（horizon）。

图 1 的主要问题在于，当优化变量的维度较高时，需要学习的参数过多（标准的 LSTM，一个隐藏单元包含 8 个全连接层）。因此，常见的做法是，采用 Coordinate-wise 的架构，如图 2 所示，优化变量 $\mathbf{x}_ t$ 的每一个分量的更新单独使用一个 LSTM 实现——

![Coordinate-wise LSTM-based L2O](C:\Users\Jhon Hu\iCloudDrive\Documents\Blog\jhonhu1994.github.io-main\images\Learningtooptimize\Coordinated-wise_LSTM-based_L2O.png)

<center><p><font size="3"><em>Fig 2. Coordinate-wise LSTM-based L2O (one step)</em></font><br/></p></center>

同时为了进一步减少网络的参数数量，所有的 LSTM 共享权值，每个优化分量的不同行为通过各自的激活函数实现。

采用 Coordinate-wise 的架构允许我们使用更小的网络，但其忽略了每个分量之间的联系[^1]；权值共享在减少参数数量的同时使得优化器对于优化分量的顺序具有不变性，但其忽略了不同分量可能存在的差异性。对于后者，可以根据优化问题的结构，对优化变量进行分组，不同组采用不同的网络参数，组内所有分量则权值共享。对于前者，可以考虑采用 Hierarchical 的网络架构，如图 3 所示：

![Hierarchical LSTM-based L2O](C:\Users\Jhon Hu\iCloudDrive\Documents\Blog\jhonhu1994.github.io-main\images\Learningtooptimize\Hierarchical_LSTM-based_L2O.png)

<center><p><font size="3"><em>Fig 3. Hierarchical LSTM-based L2O</em></font><br/></p></center>

最底层的 Parameter RNN 仍然实现 Coordinate-wise 的更新；中间层的 Tensor RNN 接受一簇 Parameter RNN（根据问题结构划分）输出值的平均，其输出再加入 Parameter RNN 作为偏差项，用于提取簇内分量的依赖性；最顶层的 Global RNN 接受所有 Tensor RNN 输出值的平均，其输出再加入 Tensor RNN 作为偏差项，用于提取簇间的依赖性。通过在底层使用很少量的隐藏元，Hierarchical 架构实现了更低的内存和计算开销。

LSTM-based L2O 面临的一个主要困境是，受限于深层网络训练的困难性，实际中网络的展开长度不能设置过高（10~20 左右）。那么对于一个需要较多步迭代的问题，整个优化轨迹就必须被划分为连续的较短片段，每个片段调用 LSTM 优化器进行优化。这会导致训练好的优化程序在测试时表现出不稳定性并产生低质量的解（截断偏差），如图 4 中黄色曲线所示：

![The Behavior of different Optimizers](C:\Users\Jhon Hu\iCloudDrive\Documents\Blog\jhonhu1994.github.io-main\images\Learningtooptimize\Behavior_of_Optimizers.png)

<center><p><font size="3"><em>Fig 4. The Behavior of different Optimizers</em></font><br/></p></center>

初步的解决方法是对网络的训练过程进行优化，比如采用渐进式的训练方案，逐步地增加LSTM网络地展开长度，以缓解截断偏差和梯度消失/爆炸之间的 LSTM-based L2O 困境。

[^1]: 通过利用每个分量之间的联系，往往的确可以加快算法收敛速度，如拟Newton法就比梯度下降法快。

### 2. RL-based L2O

此方法

<img src="C:\Users\Jhon Hu\iCloudDrive\Documents\Blog\jhonhu1994.github.io-main\images\Learningtooptimize\Unconstrained_continuous_optimization.png"  width=70%/>

#### 1) 问题

针对某个无约束优化问题 $\min _{\mathbf{x}\in\mathcal{X}}\; f(\mathbf{x})$ ，学习最优的一阶优化算法，即
$$
\mathbf{x}_ {t+1}=\mathbf{x}_ t+\mathbf{F}\left(f(\mathbf{x}_ 1),\cdots,f(\mathbf{x}_ n);\,\nabla f(\mathbf{x}_ 1),\cdots,\nabla f(\mathbf{x}_ n)\right).
$$

#### 2) 方法

优化算法的执行可视为马尔可夫决策过程（Markov Decision Process, MDP）中的一个策略的执行，目标就是找到最优的策略（任何一个特定的优化算法，例如梯度下降、共轭梯度法、拟Newton法等，都是一个（确定性）策略；所有的一阶优化算法组成了策略空间）。因此，可以将其建模为一个具有连续动作空间和状态空间的强化学习任务（Policy-based），即
$$
\mathbf{F}\left(f(\mathbf{x}_ 1),\cdots,f(\mathbf{x}_ n);\,\nabla f(\mathbf{x}_ 1),\cdots,\nabla f(\mathbf{x}_ n)\right)=\pi(s_ n)
$$
其中，状态 $s_ n=\{\mathbf{x}_ n,\nabla f(\mathbf{x}_ n),\cdots,\nabla f(\mathbf{x}_ {n-T+1}),f(\mathbf{x}_ n),\cdots,f(\mathbf{x}_ {n-T+1})\}$，动作空间是所有的一阶更新步（负梯度、共轭梯度、拟 Newton步、Someone Else），惩罚（奖励）函数为 $c(s_ n)=f(\mathbf{x}_ n)$。策略 $\pi$ （的均值）使用两层前馈神经网络参数化：隐藏层包含50个神经元，使用Softplus 激活函数；输出层使用线性激活函数；网络输入状态信息时排除 $\mathbf{x}_ n$​；使用引导策略搜索方法（Guided Policy Search）训练网络。

## Model-Based L2O

Model-Free L2O 使用通用的神经网络架构实现优化变量的迭代更新，不参考任何已有的算法，因此具有发现全新算法的可能。其缺点是缺乏理论上的收敛保证和可解释性，同时目前也只用于无约束的简单情形（很难在损失函数中对约束条件进行量化）。与之相对的，Model-Based L2O 选择某个现存的解析算法作为学习的基础构建特定的神经网络架构，网络的特定层、连接关系和激活函数都是对此解析算法的某个操作的模拟。

### 1. Algorithm Unfolding

Algorithm Unfolding 或 Deep Unfolding 的思想起源是很具有启发性的。考虑 Lasso 问题：
$$
\min_ {\mathbf{x}\in\mathbb{R}^ n}\;\frac{1}{2}\lVert\mathbf{y}-\mathbf{W}\mathbf{x}\rVert_ 2^2+\theta\lVert\mathbf{x}\rVert_ 1^ 1
$$
其中，

### 2. 获取函数



## 总结

最后总结一下贝叶斯方法的优势和劣势。

## 相关文献

[1] Shahriari B, Swersky K, Wang Z, et al. Taking the human out of the loop: A review of Bayesian optimization[J]. Proceedings of the IEEE, 2016, 104(1): 148-175.

[2] Frazier P I. A tutorial on Bayesian optimization[J]. arXiv preprint arXiv:1807.02811, 2018.

[3] Snoek J, Larochelle H, Adams R P. Practical bayesian optimization of machine learning algorithms[J]. Advances in neural information processing systems, 2012, 25.

[JhonHu]:    https://jhonhu1994.github.io  "JhonHu"
