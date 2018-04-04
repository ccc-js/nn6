const nn6 = require('../lib/nn6')

// 2x+y=3
//  x+y=2
// 解答： x=1, y=1
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {x, y, r1, r2, _2y, f} = this.addVariables(['x', 'y', 'r1', 'r2', '_2y', 'f'])
    this.setDumpVariables(['x', 'y', 'r1', 'r2', 'f'])
    let [c1, c2, c3] = this.newConstants([1, 2, 3])
    this.addInputs([[x, c1], [y, c2]])
    this.setOutput(f)
    this.addGates([
      new nn6.gate.Mul([c2, y], _2y),
      new nn6.gate.Add([x, _2y], r1),
      new nn6.gate.Add([x, y], r2),
      new nn6.net.SqrtError([[r1, r2], [c3, c2]], f, this)
    ])
  }
}

nn6.GradientDescendent(new Net1(), 1000)
