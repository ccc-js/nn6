const G = require('./gate')
const N = require('./net')

class Mlp extends N.NetGate {
  constructor(inputs, hSize, out, net) {
    super(inputs, out, net)
    let hidden = net.newVector(hSize)
    this.gates = [
      new N.Connect([inputs], hidden, net, -0.2, 0.2),
      new N.Connect([hidden], out, net, -2.0, 2.0)
    ]
  }
}

class Perceptron2 extends N.Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    let x = [x1, x2]
    this.inputs = [ x ]
    this.out = [ f ]
    this.gates = [
      new N.Neuron([x], f, this)
    ]
  }
}

class Mlp2 extends N.Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    this.inputs = [ x1, x2 ]
    this.out = [ f ]
    this.gates = [
      new Mlp(this.inputs, 3, this.out, this)
    ]
  }
}

module.exports = {
  Mlp,
  Perceptron2,
  Mlp2,
}