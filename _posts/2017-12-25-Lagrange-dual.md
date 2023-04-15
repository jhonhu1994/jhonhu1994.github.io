---
layout: post
title: 最优化理论之 Lagrange 对偶 
description: 这里想谈一谈Lagrange 对偶理论。
category: blog
---

这里想谈一谈 Lagrange 对偶理论。

在最优化领域，特别是凸优化中，Lagrange 对偶是极其重要的概念。对于一个优化问题：

$$\min_ {\mathbf{x}\in\mathbb{R}^n}\quad f(\mathbf{x})\qquad\qquad\qquad\qquad\quad\\
\mathrm{s.t.}\quad h_ i(\mathbf{x})=0,\,i\in\mathcal{E}=\{1,2,\cdots,l\},\\
\qquad\quad h_ i(\mathbf{x})\geq0,\,i\in\mathcal{I}=\{l+1,\cdots,m\}.\tag{1}$$

定义其 Lagrangian 函数为

$$L(\mathbf{x},\pmb{\lambda})=f(\mathbf{x})-\pmb{\lambda}^\mathrm{T}\mathbf{h}(\mathbf{x})=f(\mathbf{x})-\sum_ {i=1}^{m}\lambda_ ih_ i(\mathbf{x}),\tag{2}$$

其中，$\lambda_ i$ 为 Lagrangian 乘子（或称为对偶变量）且 $\lambda_ i\geq 0,\forall i\in\mathcal{I}$。由 Lagrangian 函数的定义可知，对于任意 $\mathbf{x}\in\mathcal{D}=\{\mathbf{x}\mid h_ i(\mathbf{x})=0,\forall i\in\mathcal{E};h_ i(\mathbf{x})\geq 0,\forall i\in\mathcal{I}\}$，有 $L(\mathbf{x},\pmb{\lambda})$ 是关于 $\pmb{\lambda}$ 的仿射函数。

### 原始问题

考虑关于 $\mathbf{x}$ 的函数：

$$\theta_ \mathrm{P}(\mathbf{x})=\max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;L(\mathbf{x},\pmb{\lambda}).\tag{3}$$

显然，对于 $\theta_ \mathrm{P}(\mathbf{x})$，当 $\mathbf{x}\notin\mathcal{D}$，即存在某个 $i\in\mathcal{E}$ 使得 $h_ i(\mathbf{x})\neq 0$ 或存在某个 $i\in\mathcal{I}$ 使得 $h_ i(\mathbf{x})<0$，有 $\theta_ \mathrm{p}(\mathbf{x})=+\infty$；当 $\mathbf{x}\in\mathcal{D}$，则有 $\theta_ \mathrm{P}(\mathbf{x})=f(\mathbf{x})$。因此，问题 (1) 等价于 Lagrangian 函数的极小极大问题：

$$\min_ {\mathbf{x}\in\mathbb{R}^n}\;\theta_ \mathrm{P}(\mathbf{x})=\min_ {\mathbf{x}\in\mathbb{R}^n}\;\max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;L(\mathbf{x},\pmb{\lambda}).\tag{4}$$

一般称极小极大问题为原始问题（primal question）。

### 对偶问题

考虑极小极大问题 (4) 的对偶问题（dual problem），即极大极小问题：

$$\max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;\theta_ \mathrm{D}(\pmb{\lambda})=\max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;\min_ {\mathbf{x}\in\mathbb{R}^n}\;L(\mathbf{x},\pmb{\lambda}),\tag{5}$$

其中 $\theta_ \mathrm{D}(\pmb{\lambda})=\inf_ {\mathbf{x}\in\mathbb{R}^n}\,L(\mathbf{x},\pmb{\lambda})$ 被称为对偶函数。显然，$\theta_ \mathrm{D}(\pmb{\lambda})$ 为凹函数（仿射函数的逐点下确界），因此，对偶问题是一个凸优化问题。

### 对偶间隔和强对偶性

结合式 (3) 和 (5)，有

$$\theta_ \mathrm{D}(\pmb{\lambda})=\min_ {\mathbf{x}\in\mathbb{R}^n}\;L(\mathbf{x},\pmb{\lambda})\leq L(\mathbf{x},\pmb{\lambda})\leq \max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;L(\mathbf{x},\pmb{\lambda})=\theta_ \mathrm{P}(\mathbf{x}),\tag{6}$$

即 $\theta_ \mathrm{D}(\pmb{\lambda})\leq\theta_ \mathrm{P}(\mathbf{x})$，则当然有

$$d^\ast=\max_ {\pmb{\lambda};\lambda_ i\geq 0,i\in\mathcal{I}}\;\theta_ \mathrm{D}(\pmb{\lambda})\leq\min_ {\mathbf{x}\in\mathbb{R}^n}\;\theta_ \mathrm{P}(\mathbf{x})=p^\ast.\tag{7}$$

也就是说，对偶问题的最优值 $d^\ast$（极大）一定不大于原始问题的最优值 $p^\ast$（极小）。一个直接的推论是——

_设_ $\mathbf{x}_ 0$ _和_ $\pmb{\lambda}_ 0$ _分别是原始问题  (4) 和对偶问题 (5) 的可行点，当_ $\theta_ \mathrm{D}(\pmb{\lambda})=\theta_ \mathrm{P}(\mathbf{x})$ _时，有_ $\mathbf{x}_ 0$ _和_ $\pmb{\lambda}_ 0$ _分别为原始问题 (4) 和对偶问题 (5) 的最优解_  $\mathbf{x}^\ast$ _和_ $\pmb{\lambda}^\ast$。

注意，上述推论的逆定理并不成立，即求得原始问题 (4) 和对偶问题 (5) 的最优解 $\mathbf{x}^\ast$ 和 $\pmb{\lambda}^\ast$，并不一定有 $\theta_ \mathrm{D}(\pmb{\lambda}^\ast)=\theta_ \mathrm{P}(\mathbf{x}^\ast)$，只能确定有 $\theta_ \mathrm{D}(\pmb{\lambda}^\ast)\leq\theta_ \mathrm{P}(\mathbf{x}^\ast)$。称这种性质为弱对偶性，相应的 $\Delta=\theta_ \mathrm{P}(\mathbf{x}^\ast)-\theta_ \mathrm{D}(\pmb{\lambda}^\ast)$ 被称为对偶间隔。 当满足 $\theta_ \mathrm{P}(\mathbf{x}^\ast)=\theta_ \mathrm{D}(\pmb{\lambda}^\ast)$ 时，称为强对偶性成立，此时可以用解对偶问题代替求解原始问题（对偶问题为凸优化问题）。

可以证明，对于一个凸优化问题，当其存在一个严格可行点 $\mathbf{x}_ 0$（对所有 $i\in\mathcal{I}$，满足 $h_ i(\mathbf{x}_ 0)>0$ ）时，强对偶性成立（Slater定理）。显然，等式（仿射函数）约束的凸优化问题满足强对偶性；无约束凸优化问题更然（无约束优化问题的对偶问题是其本身）。

### 强对偶性与 KKT 条件

当强队对偶性成立，即对偶间隔为 $0$，有

$$f(\mathbf{x}^\ast)=\theta_ \mathrm{P}(\mathbf{x}^\ast)=\theta_ \mathrm{D}(\pmb{\lambda}^\ast)=\min_ {\mathbf{x}\in\mathbb{R}^n}\;L(\mathbf{x},\pmb{\lambda}^\ast)\leq L(\mathbf{x}^\ast,\pmb{\lambda}^\ast)\leq f(\mathbf{x}^\ast),\tag{8}$$

最后一个不等号来源于 $\lambda_ i^\ast\geq 0,h_ i(\mathbf{x}^\ast)\geq 0,\;\forall i\in\mathcal{I}$。显然，式 (8) 中所有小于等于号全取等号，即

$$f(\mathbf{x}^\ast)=\theta_ \mathrm{P}(\mathbf{x}^\ast)=\theta_ \mathrm{D}(\pmb{\lambda}^\ast)=\min_ {\mathbf{x}\in\mathbb{R}^n}\;L(\mathbf{x},\pmb{\lambda}^\ast)= L(\mathbf{x}^\ast,\pmb{\lambda}^\ast)= f(\mathbf{x}^\ast).\tag{9}$$

根据式 (9)，最优解 $(\mathbf{x}^\ast,\pmb{\lambda}^\ast)$ 满足下列 KKT（Karush-Kuhn-Tucker）方程[^1]：

$$\nabla_ \mathbf{x} L(\mathbf{x}^\ast,\pmb{\lambda}^\ast) = \mathbf{0},\qquad\qquad\qquad\,\\
h_ i(\mathbf{x}^\ast) = 0,\;\forall i\in\mathcal{E},\;\;\,\\
h_ i(\mathbf{x}^\ast) \geq 0,\;\forall i\in\mathcal{I},\;\;\,\\
\;\;\lambda_ i^\ast \geq 0,\;\forall i\in\mathcal{I},\\
\lambda_ i^\ast h_ i(\mathbf{x}^\ast) = 0,\;\forall i\in\mathcal{E}\cup\mathcal{I},\tag{10}$$

其中， $\nabla_ \mathbf{x} L(\mathbf{x}^\ast,\pmb{\lambda}^\ast) = \mathbf{0}$ 被称为零导数条件，来源于 $L(\mathbf{x}^\ast,\pmb{\lambda}^\ast)=\min_ {\mathbf{x}\in\mathbb{R}^n}\;L(\mathbf{x},\pmb{\lambda}^\ast)$； $\lambda_ i^\ast h_ i(\mathbf{x}^\ast) = 0,\;\forall i$ 被称为互补松弛（complementary slackness）条件，来源于 $L(\mathbf{x}^\ast,\pmb{\lambda}^\ast)= f(\mathbf{x}^\ast)$。

[^1]: KKT 方程是带约束优化问题中的一个很一般的理论，其在一些温和的约束规格（constraint qualification）下都成立，这里的强对偶性即是一例（属于比较强的条件）。

### 对偶的本质？

对 engineers 而言，理解 Lagrange 对偶方法，上述内容已然足够。不过，一个 researcher 可能会指出 ，上述内容回避了一个重要的问题——Lagrangian 函数的表达式是如何导出的。此外，对于非凸问题中的对偶间隔现象也没有很好的解释 。事实上，如果想真正理解优化问题中 “对偶（duality）” 的本质，需要去学习数学家   Werner Fenchel 关于共轭函数（Conjugate Function） 的研究 。可惜，中文互联网上，鲜有相关的讨论。有时间，我来填坑。

[Jhonhu]:    https://jhonhu1994.github.io  "JhonHu"
