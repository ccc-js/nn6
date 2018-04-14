const nn6 = require('../lib/nn6')

class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    let x = [ x1, x2 ]
    this.inputs = x
    this.out = [ f ]
    this.gates = [
      new nn6.net.Neuron([x], f, this)
    ]
  }
}

nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [0], [0], [1]], 1001) // and
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]], 1000) // or
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 1000) // xor

