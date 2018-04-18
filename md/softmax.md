# Softmax 函数的特点和作用是什么？ 

-- https://www.zhihu.com/question/23765351

注意看 《忆臻》那一段！

形式非常简单，这说明我只要正向求一次得出结果，然后反向传梯度的时候，只需要将它结果减1即可，后面还会举例子！

Softmax Layer -- https://www.jianshu.com/p/cb93d5e39bca

1.前向传播
该过程比较简单，对输入的每个节点进行softmax（x）计算。但是需要注意的是，
由于存在指数函数exp，对于输入很大的实数会softmax数值越界，导致预想不到的结果。
所以在做softmax之前，需要将数据做简单的预处理，即：找出输入节点的最大值，
然后让每个节点减去该最大值，使得输入节点都是小于等于0的，这样就能避免数值越界。

2.反向传播
softmax层的导数要分两种情况：1）该节点是输出类别 (i  =  j);2）该节点不为输出类别（i != j）


Softmax 函数的特点和作用是什么？ -- https://www.zhihu.com/question/23765351

当我们对分类的Loss进行改进的时候，我们要通过梯度下降，每次优化一个step大小的梯度
我们定义选到yi的概率是

然后我们求Loss对每个权重矩阵的偏导，应用链式法则（中间推导省略）。

最后结果的形式非常的简单，只要将算出来的概率的向量对应的真正结果的那一维减1，就可以了

当我们对分类的Loss进行改进的时候，我们要通过梯度下降，每次优化一个step大小的梯度

我们定义选到yi的概率是

现在看起来是不是感觉复杂了，居然还有累和，然后还要求导，每一个a都是softmax之后的形式！
但是实际上不是这样的，我们往往在真实中，如果只预测一个结果，那么在目标中只有一个结点
的值为1，比如我认为在该状态下，我想要输出的是第四个动作（第四个结点）,那么训练数据的
输出就是a4 = 1,a5=0,a6=0，哎呀，这太好了，除了一个为1，其它都是0，那么所谓的4
求和符合，就是一个幌子，我可以去掉啦！为了形式化说明，我这里认为训练数据的真实输出
为第j个为1，其它均为0！那么Loss就变成了,累和已经去掉了，太好了。现在我们要
开始求导数了！


形式非常简单，这说明我只要正向求一次得出结果，然后反向传梯度的时候，
只需要将它结果减1即可，后面还会举例子！


這也是為何 recurrent.js 沒有實作 softmax 的梯度之原因。

https://github.com/karpathy/recurrentjs/blob/master/src/recurrent.js


```
  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(var i=0,n=m.w.length;i<n;i++) { 
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
    }

    ...
  https://github.com/karpathy/recurrentjs/blob/master/character_demo.html

  for(var i=-1;i<n;i++) {
    // start and end tokens are zeros
    var ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
    var ix_target = i === n-1 ? 0 : letterToIndex[sent[i+1]]; // last step: end with END token
    lh = forwardIndex(G, model, ix_source, prev);
    prev = lh;
    // set gradients into logprobabilities
    logprobs = lh.o; // interpret output as logprobs
    probs = R.softmax(logprobs); // compute the softmax probabilities
    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);
    // write gradients into log probabilities
    logprobs.dw = probs.w;      // 梯度幾乎和前一層一樣！
    logprobs.dw[ix_target] -= 1 // 就是在這裡減一的。
  }
  var ppl = Math.pow(2, log2ppl / (n - 1));
  return {'G':G, 'ppl':ppl, 'cost':cost};

```


/*
  SoftmaxLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(1, 1, this.out_depth, 0.0);

      // compute max activation
      var as = V.w;
      var amax = V.w[0];
      for(var i=1;i<this.out_depth;i++) { // 取得最大的 output
        if(as[i] > amax) amax = as[i];
      }

      // compute exponentials (carefully to not blow up)
      var es = global.zeros(this.out_depth);
      var esum = 0.0;
      for(var i=0;i<this.out_depth;i++) {
        var e = Math.exp(as[i] - amax); // 減掉最大的 output ，避免爆掉
        esum += e;
        es[i] = e;
      }

      // normalize and output to sum to one
      for(var i=0;i<this.out_depth;i++) {
        es[i] /= esum;
        A.w[i] = es[i];
      }

      this.es = es; // save these for backprop
      this.out_act = A;
      return this.out_act;
    },
    backward: function(y) {

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

      for(var i=0;i<this.out_depth;i++) {
        var indicator = i === y ? 1.0 : 0.0;
        var mul = -(indicator - this.es[i]);
        x.dw[i] = mul;
      }

      // loss is the class negative log likelihood
      return -Math.log(this.es[y]);
    },
*/

/*
Softmax 的梯度 https://github.com/dritchie/adnn/blob/master/ad/functions.js
// http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
fns.tensor.softmax = func.newUnaryFunction({
	OutputType: Tensor,
	name: 'tensor.softmax',
	forward: function(t) {
		return t.softmax();
	},
	backward: function(t) {
		// For each input entry, accumulate partial derivatives
		//    for each output entry
		var n = t.dx.data.length;
		var s = 0;
		for (var i = 0; i < n; i++) {
			s += this.x.data[i] * this.dx.data[i];
		}
		for (var j = 0; j < n; j++) {
			t.dx.data[j] += this.x.data[j] * (this.dx.data[j] - s);
		}
	}
});
*/

// Good ! 深度学习基础 (九)--Softmax (多分类与评估指标) -- https://testerhome.com/topics/11262
// (X) Softmax Regression -- https://blog.csdn.net/u012328159/article/details/72155874
// 一天搞懂深度學習 -- https://www.slideshare.net/tw_dsconf/ss-62245351
// 一天搞懂深度學習--學習心得 -- https://www.youtube.com/watch?v=ZrEsLwCjdxY
// 一日搞懂生成式對抗網路 -- https://www.slideshare.net/tw_dsconf/ss-78795326
/*

Softmax 的梯度 https://github.com/dritchie/adnn/blob/master/ad/functions.js
// http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
fns.tensor.softmax = func.newUnaryFunction({
	OutputType: Tensor,
	name: 'tensor.softmax',
	forward: function(t) {
		return t.softmax();
	},
	backward: function(t) {
		// For each input entry, accumulate partial derivatives
		//    for each output entry
		var n = t.dx.data.length;
		var s = 0;
		for (var i = 0; i < n; i++) {
			s += this.x.data[i] * this.dx.data[i];
		}
		for (var j = 0; j < n; j++) {
			t.dx.data[j] += this.x.data[j] * (this.dx.data[j] - s);
		}
	}
});
*/

