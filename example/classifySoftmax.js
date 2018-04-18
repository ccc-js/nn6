const nn6 = require('../lib/nn6')

class Net1 extends nn6.net.Net {
  constructor () {
    super()
    // this.step = 0.001
    // this.moment = 0.00001 // 0.01
    let {a,b,c,d,e,f,x,y} = this.addVariables(['a', 'b', 'c', 'd', 'e', 'f', 'x', 'y'])
    this.setDumpVariables([ 'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y'])
    this.inputs = [ a, b, c, d, e, f ]
    let o = this.newVector(2)
    this.out = [ x, y ]
    this.gates = [
      // new nn6.mlp.Mlp(this.inputs, [5], o, this), // 成功
      // new nn6.mlp.Mlp(this.inputs, [], o, this, nn6.gate.LeakyRelu),
      // 問題是，當所有的輸出值都小於 0，就會都被 Relu 截掉，於是變成 [0,0,0....]，這個稱為 dying ReLU problem 
      // 使用 Relu 有時成功，有時失敗，改使用 LeakyRelu 就不會有這個問題了。
      new nn6.mlp.Mlp(this.inputs, [5], o, this, nn6.gate.LeakyRelu),
      new nn6.net.SoftmaxLayer(o, this.out)
    ]
    this.fLoss = nn6.loss.crossEntropy
  }
}

const inputs = [
  [1,1,1,0,0,0],
  [1,0,1,0,0,0],
  [1,1,1,0,0,0],
  [0,0,1,1,1,0],
  [0,0,1,1,0,0],
  [0,0,1,1,1,0],
]

const outs = [
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1],
  [0, 1]
]

nn6.gradientLearning(new Net1(), inputs, outs, 5000, 100)

