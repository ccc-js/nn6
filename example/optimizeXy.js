const nn6 = require('../lib/nn6')

// optimize: f = x^2 + y^2
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {x, y, x2, y2, f} = this.addVariables(['x', 'y', 'x2', 'y2', 'f'])
    this.setDumpVariables(['x', 'y', 'f'])
    // let [c1, c2, c3] = nn6.node.newConstants([1, 2, 3])
    this.out = f
    this.gates = [
      new nn6.gate.Mul([x, x], x2),
      new nn6.gate.Mul([y, y], y2),
      new nn6.gate.Add([x2, y2], f),
    ]
  }
}

nn6.gradientDescendent(new Net1(), 50)
