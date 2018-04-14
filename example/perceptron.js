const nn6 = require('../lib/nn6')

// 2x+y=3
//  x+y=2
// 解答： x=1, y=1
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {x1, x2, w0, w1, w2, s, f} = this.addVariables(['x1', 'x2', 'w0', 'w1', 'w2', 's', 'f'])
    this.setDumpVariables(['x1', 'x2', 'w0', 'w1', 'w2','s', 'f'])
    let [n1] = nn6.node.newConstants([-1])
    this.inputs = [ x1, x2 ]
    this.out = [ f ]
    this.gates = [
      new nn6.net.VDot([[n1, x1, x2], [w0, w1, w2]], s, this),
      new nn6.gate.Sigmoid([s], f, this),
    ]
  }
}

nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [0], [0], [1]], 201) // and
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]], 100) // or
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 100) // xor

