const nn6 = require('../lib/nn6')

// 2x+y=3
//  x+y=2
// 解答： x=1, y=1
class Graph1 extends nn6.Graph {
  constructor () {
    super()
    let {x, y, r1, r2, e1, e2, f} = this.addVariables(['x', 'y', 'r1', 'r2', 'e1', 'e2', 'f'])
    let {c1, c2, c3} = nn6.node.newConstants(1, 2, 3)
    this.addInputs([[x, y], [c1, c2], new nn6.Layer())
    this.setOutput(f)
    this.addGates([
      new nn6.Layer([[x, y], [c2, c1]], [r1, r2], nn6.gate.Mul),
      new nn6.Layer([[r1, r2], [c3, c2]], [e1, e2], nn6.gate.Sub),
      new nn6.Layer([[e1, e2], [e1, e2]], [e1s, e2s], nn6.gate.Mul),
      new nn6.Sum([e1s, e2s], f),
    ])
    console.log('vars=%j', this.vars)
  }
}

let net = new nn6.Net(new Graph1())
net.run(300)
