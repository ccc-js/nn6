const G = require('./gate')
const N = require('./net')

class Mlp extends N.NetGate {
  constructor(inputs, hSizes, out, net, Sig=G.Sigmoid) {
    super(inputs, out, net)
    let hLen = hSizes.length
    let hiddens = [ inputs ]
    for (let i=0; i<hLen; i++) {
      hiddens[i+1] = net.newVector(hSizes[i])
      this.gates.push(new N.Connect([ hiddens[i] ], hiddens[i+1], net, -0.2, 0.2, Sig))
    }
    this.gates.push(new N.Connect([ hiddens[hLen] ], out, net, -2.0, 2.0, Sig))
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
      new Mlp(this.inputs, [3], this.out, this)
    ]
  }
}

module.exports = {
  Mlp,
  Perceptron2,
  Mlp2,
}