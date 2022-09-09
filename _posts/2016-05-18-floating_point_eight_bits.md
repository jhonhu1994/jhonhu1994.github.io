---
layout: post
title: 8位浮点数
description: 大部分时候，
category: blog
---

## 定点数与浮点数

严格上讲，计算机存储的只能是固定位数的整数。对于含有分数部分的数，计算机无法识别 “.” ，目前的处理方式有定点（fixed-point）和浮点（floating-point）两种。

假设现在可用一个字节（8比特）存储一个数（比如 $5.5_ {(10)}$ ）。对于定点数，顾名思义，即 "." 始终固定在给定的存储位的同一位置。图1给出了使用（8，4）定点数表示 $5.5_ {(10)}$ 的示意图：

![5.5 in (8,4) fixed point](/images/floatingpoint8/fixed_point_5_5.png)

形式上看，"." 将8个比特位划分为两部分，"." 左边的位表示整数部分，"." 右边的位表示分数部分。实际存储时，即用两个半字节整数表示一个定点数，只不过小数点后的第一位是半位，下一位是四分之一位，下一位是八分之一位，以此类推。定点数的形式简洁，也易于硬件实现，其缺点在于表示范围受限。以8位定点数为例，上图的（8，4）形式可表示的最大正数为 $2^3 = 8_ {(10)}$ ，即便采用（8，1）的形式也只可以表示到 $2^6=64_ {(10)}$ 。大多数程序在计算过程中需要使用范围更广的数字，因此定点数在当今的计算世界并不常用。

浮点数的思想，本质上可以视为2进制的科学计数法。还是以 $5.5_ {(10)}$ 为例，其二进制形式为 $101.1_ {(2)}$ ，转换为科学计数法即为 $1.011_ {(2)}\times 2^{\color{red} 2}$ ；遵循科学计数法的命名，$1.011_ {(2)}$ 称为尾数（mantissa）或有效数，$\color{red} 2$ 称为指数（exponent）。所以，给定存储位数，浮点数格式可以由以下三个要素定义（以8位浮点数为例）：

![FP8-(1,4,3)](/images/floatingpoint8/floating_point_FP8.png)

<center><p><font size="3">FP8 - (1, 4, 3)</font><br/></p></center>

其中，第1位为符号位（"0"表示正数，"1"表示负数）；中间的 $e=4$ 个比特表示 biased exponent，即 $\mbox{exponent}+(2^{s-1}-1)$（引入bias的目的是在不引入补码的情况下允许负指数，也就是4比特指数的表示范围为 $-7\sim 8$ ）；最后 $m=3$ 个比特表示mantissa的小数部分。以 $5.5_ {(10)}=1.011_ {(2)}\times 2^2$ 为例，将其表示为 FP8-(1,4,3) 的形式，则符号位为0，四位 biased exponent 为 $2+7=9_ {(10)}=1001_ {(2)}$ ， mantissa 为 $1.011$ 小数点后的三位 $011$ (对于二进制计数法，小数点前面一定是1，所以可以舍掉以节省空间[^1]）。相比于定点数，取决于 exponent 的取值，浮点数允许 "." 在给定存储位的“任何”位置上浮动，由此得名。

 来看 FP8-(1,4,3) 可以表示的最大正数为 $1.111_ {(2)}\times 2^{15-7}=111100000_ {(2)}=480_ {(10)}$  ；可以表示的最小正数为 $1.000_ {(2)}\times 2^{0-7}=2^{-7}\approx 0.0078_ {(10)}$ 。显然，相比于（8，4）定点数（不考虑零点，最大可表示正数为 $2^3=8_ {(10)}$ ，最小可表示正数为 $2^{-4}=0.0625_ {(10)}$ ），浮点数可以处理更宽范围内的数。然而，不要忘了，我们始终还是只有8个存储位；换言之，最多也只能表示 $2^8=256$ 个不同的值。下图分别罗列了（8，4）定点数（上图）和 FP8-(1,4,3) 浮点数（下图）所表示的256个数字在数轴上的位置：

![Distribution of values for (8,4) fixed point](/images/floatingpoint8/distribution_of_values_for_fixed_point.png)

![Distribution of values for (1,4,3) floating point](/images/floatingpoint8/distribution_of_values_for_floating_point.png)

可以注意到，定点数是均匀的，其相邻两数的间隔固定为 $2^{-4}=0.0625$ ；而浮点数却并不是均匀分布的，相反，相邻两数之间的差值随着远离零点越来越大。所以，信息密度的不均匀正是使得浮点数得以处理更大范围的数值的原因。而从工程角度讲，这种不均匀的信息密度也是有道理的——保证相对误差较小。

最后给出将一个十进制实数转换（量化）为浮点数的具体过程（注意，尾数的舍入规则是保证 mantissa 最后一位是0，从而使得一半的数向上取”整“，一半的数向下取”整“）：

![Converting a real number to a floating point number](/images/floatingpoint8/converting_real_to_floating_point.png)

## IEEE 754标准

[POSIX](https://zh.wikipedia.org/zh-sg/POSIX)对行的[定义](http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_206)如下：

  > 3.206 Line

  > A sequence of zero or more non- <newline\> characters plus a terminating <newline\> character.

  > 行是由0个或者多个非 "换行" 符的字符组成，并且以 "换行" 符结尾。

这样做有什么好处呢，举个例子：

    //hello.c
    #include head.h
    print('hello')
    
    //world.c
    #include tail.h
    print('hello')

如果这两个文件都按 POSIX 规范来写， 在`cat *.c`之后，是没有问题的：

    //cat.c
    
    #include head.h
    print('hello')
    #include tail.h
    print('hello')

如果缺少最后一行的换行符（如 Windows 文件那样的定义），`cat`之后，就有问题了：

    //error.c
    
    #include head.h
    print('hello')#include tail.h
    print('hello')

所以，从这点去理解 POSIX 对行的定义，非常合理，对于任意文件的拼接，也各自保持了文件的完整性。

不遵守标准带来的则是：在一些编辑器下面，比如 Sublime，他把`\n`的当做了行之间的分隔符，于是文件最后一行的`\n`就看上去成了一个新的空行，这就是错误解读标准造成的，拼接文件时也会产生不必要的麻烦，比如上例。

## \ No new line at end of file

基于上面的原因，再去看 git diff 的`\ No new line at end of file`信息，就很好解释了。

各编辑器对于换行符的理解偏差，导致的文件确实发生了变化，多了或少了最后的`0a`，那么对于 diff 程序来说，这当然是不可忽略的，但因为`0a`是不可见字符，并且是长久以来的历史原因，所以 diff 程序有个专门的标记来说明这个变化，就是：

`\ No new line at end of file`

各编辑器也有相应的办法去解决这个问题，比如 Sublime，在`Default/Preferences.sublime-settings`中设置：

    // Set to true to ensure the last line of the file ends in a newline
    // character when saving
    "ensure_newline_at_eof_on_save": true,

所以，请遵守规范。

[Jhonhu]:    https://jhonhu1994.github.io  "JhonHu"
