# Bug

## Gan 部分

Gan, RGan 都沒有成功

我在寫生成對抗網路 GAN 時，最後 Generator 生成的結果都會集中在單一的正確樣本上，而不會分散開來。

這樣雖然生成的結果分數是很好，但是卻完全沒有通用性。

請問有沒有人知道該如何避免 GAN 集中在單一正確樣本上的問題 ....

Gan 演算法參考 -- https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f

程式： https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py

關鍵可能在能量函數 BCELoss 上

criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

看來衡量函數應該是：Binary Cross Entropy

1. 先產生 n 個輸入向量 (vectors[]) ，產生 n 個樣本 (samples[]) ，對應到 n 個正確答案 (facts[])。
2. 預期第一個向量會產生第一個答案，第二個向量會產生第二個答案 ....
3. 衡量函數用 Binary Cross Entropy (samples[], facts[])

對於答案為實數值者，也可以採用 SquareError 。 sum((xi-yi)^2)

## 梯度下降法部分

目前測試都正確，solveXy, solveMatXy, solveLearnXy, optimizeAddTest.js, optimizeSumTest.js 等都可以正常運作並找到解答！


## 神經網路部分

單層神經網路像 perceptron.js, neuron.js 都可以正確運作。

## MLP

多層 mlp 程式，像是 mlpXor, mlp7seg 學習都無法正確學習，看來像是梯度消失 (sigmoid) 與爆炸 (tanh) 的問題，但我不知道該如何解決。

後來嘗試加入動量 moment 並增大 step 之後就可以學會了！

## RNN

Rnn01 目前運作還算OK，但不知算不算正確？

```
$ node rnn01
...
1900:
  0 => x:0.001834 f:0.207705
  1 => x:-0.008886 f:0.996248
  2 => x:1.007418 f:0.007422
  3 => x:0.001833 f:0.207674
  4 => x:-0.008884 f:0.996250
  5 => x:1.007417 f:0.007420
  6 => x:0.001833 f:0.207644
  7 => x:-0.008882 f:0.996251
  8 => x:1.007415 f:0.007419
  ==> energy = 0.041987724895997273
```

## LSTM

LSTM01 目前運作成功了，但不識每次都正確，以下是正確時候的結果：

```
$ node lstm01
...
2900:
  0 => x:0.028863 f:0.399116
  1 => x:-0.030920 f:0.965941
  2 => x:1.002323 f:0.001067
  3 => x:0.028863 f:0.399086
  4 => x:-0.030921 f:0.965969
  5 => x:1.002324 f:0.001065
  6 => x:0.028863 f:0.399057
  7 => x:-0.030922 f:0.965998
  8 => x:1.002324 f:0.001064
  ==> energy = 0.4366573645571143
```
