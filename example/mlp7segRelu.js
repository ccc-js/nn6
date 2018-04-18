const nn6 = require('../lib/nn6')

class Mlp7Seg extends nn6.net.Net {
  constructor () {
    super()
    let {a,b,c,d,e,f,g,o0,o1,o2,o3,o4,o5,o6,o7,o8,o9} = this.addVariables(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9'])
    this.setDumpVariables([ /*'a', 'b', 'c', 'd', 'e', 'f', 'g', */ 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9'])
    this.inputs = [ a, b, c, d, e, f, g ]
    let o = this.newVector(10)
    this.out = [ o0, o1, o2, o3, o4, o5, o6, o7, o8, o9 ]
    this.gates = [
      // new nn6.mlp.Mlp(this.inputs, [5], this.out, this), // 成功
      new nn6.mlp.Mlp(this.inputs, [8,10, 10], o, this, nn6.gate.LeakyRelu), // 改為 LeakyRelu 後就成功了！
      new nn6.net.SoftmaxLayer(o, this.out)
    ]
    this.fLoss = nn6.loss.crossEntropy
  }
}

const inputs = [
// A B C D E F G 
  [1,1,1,1,1,1,0], // 0
  [0,1,1,0,0,0,0], // 1
  [1,1,0,1,1,0,1], // 2
  [1,1,1,1,0,0,1], // 3
  [0,1,1,0,0,1,1], // 4
  [1,0,1,1,0,1,1], // 5
  [1,0,1,1,1,1,1], // 6
  [1,1,1,0,0,0,0], // 7
  [1,1,1,1,1,1,1], // 8
  [1,1,1,1,0,1,1], // 9
]

const outs = [
   [1,0,0,0,0,0,0,0,0,0], // 0
   [0,1,0,0,0,0,0,0,0,0], // 1
   [0,0,1,0,0,0,0,0,0,0], // 2
   [0,0,0,1,0,0,0,0,0,0], // 3
   [0,0,0,0,1,0,0,0,0,0], // 4
   [0,0,0,0,0,1,0,0,0,0], // 5
   [0,0,0,0,0,0,1,0,0,0], // 6
   [0,0,0,0,0,0,0,1,0,0], // 7
   [0,0,0,0,0,0,0,0,1,0], // 8
   [0,0,0,0,0,0,0,0,0,1], // 9
  ]

nn6.gradientLearning(new Mlp7Seg(), inputs, outs, 5000, 100)

