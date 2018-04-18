const Nd = require('./node')
const U = require('./util')
const N = require('./net')

/*
# 循環生成對抗網路

對於 GAN 而言，如果 Discriminator 只判斷好壞，那 Generator 可能會只產生單一類資訊。
(這或許是我對 GAN 的誤解，不過這個誤解導致我對 GAN 進行了下列改良 ....)

為了要讓 GAN 不會產生單一類資訊，我們將類別訊息餵給 Generator，
要求 Generator 產生的樣本，其得分是由 Generator 判斷該樣本有多符合該類別而決定的。

更進一步，可以將 Discriminator 倒數第二層的輸出，直接拉給 Generator 當輸入，
這樣等於是要求 Generator 能夠產生出讓 Discriminator 具有相同感受的樣本。

問題是，失敗了，不知為何？

找找真實程式碼吧！ 

* Tensorflow.js
  * 
  * https://wellwind.idv.tw/blog/2018/04/07/tensorflow-js-basic/

* https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
  * https://github.com/roatienza/Deep-Learning-Experiments

* http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/


*/

class Generator {
  constructor(gate) {
    this.gate = gate
  }
  generate(tag) {
    let iLen = this.gate.inputs.length
    let vector = U.repeats(iLen-tag.length, ()=>U.rand(-1.0, 1.0))
    Nd.setValues(this.gate.inputs, tag.concat(vector))
    this.gate.forward()
    let sample = Nd.getValues(this.gate.out)
    return {vector, sample}
  }
  learn(tag, vector, score) {
    Nd.setValues(this.gate.inputs, tag.concat(vector))
    this.gate.forward()
    this.gate.net.resetGrad()
    for (let o of this.gate.out) o.grad = score - o.value // 問題是：這裡的梯度資訊不見了，沒辦法引導網路方向！
    // 但長期累積的效果，是否會和有梯度一樣呢？
    this.gate.backward()
    this.gate.net.adjust()
  }
}

class Discriminator {
  constructor(gate) {
    this.gate = gate
  }
  evaluate(tag, sample) {
    Nd.setValues(this.gate.inputs, tag.concat(sample))
    this.gate.forward()
    return this.gate.out[0].value
  }
  learn(tag, sample, score) {
    let out = this.gate.out
    Nd.setValues(this.gate.inputs, tag.concat(sample))
    this.gate.forward()
    this.gate.net.resetGrad()
    for (let o of this.gate.out) o.grad = score - o.value
    // out[0].grad = score - out[0].value
    this.gate.backward()
    this.gate.net.adjust()
  }
}

class RGan extends N.Net {
  constructor(facts) {
    super()
    this.facts = facts
  }
  learn(maxLoops) {
    let g = this.generator, d=this.discriminator
    for (let loop = 0; loop < maxLoops; loop++) {
      let fact = U.randChoose(this.facts)
      let tag = fact.o
      let {vector,sample} = g.generate(tag) // Generator 產生新一輪的樣本 sample
      let score = d.evaluate(tag, sample) // Discriminator 評估樣本 sample 的分數
      if (loop % 1000===0) console.log('g:%d: %s=>%s', loop, fact.t, this.dump())
      g.learn(tag, vector, score) // Generator 根據分數調整
      if (loop % 10===0) {
        let negative = U.repeats(sample.length, ()=>U.rand(0, 1)) // 負面樣本 (純亂數)
        d.learn(tag, negative, 0) // 負面樣本盡量給 0 分
        d.learn(tag, sample, 0.3) // 產生樣本應該好一點，給一些分數
        d.learn(tag, fact.i, 1)   // 標準答案最好，給滿分
        if (loop % 1000===0) console.log('  d:%d: %s=>%s', loop, fact.t, this.dump())
      }
    }
  }
  print(tag, vector, sample, score) {
    console.log('  t=%j v=%j x=%j s=%d', tag, vector, sample, score)
  }
}

module.exports = {
  Discriminator,
  Generator,
  RGan,
}