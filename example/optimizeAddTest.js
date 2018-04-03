const nn6 = require('../lib/nn6')

// f(x,y) = x^2 + 2xy + y^2 + 2
// 在 x=0 y=0 時，有最小值 2
class Graph1 extends nn6.Graph {
  constructor () {
    super()
    let v = this.addVariables(['x', 'y', 'xx', 'yy', 'f'])
    let [c2] = nn6.node.newConstants([2])
    this.addInputs([v.x, v.y])
    v.x.value = 1
    v.y.value = 2
    this.setOutput(v.f)
    this.addGates([
      new nn6.gate.Mul([v.x, v.x], v.xx),
      new nn6.gate.Mul([v.y, v.y], v.yy),
      new nn6.gate.Add([v.xx, v.yy], v.f)
    ])
    console.log('vars=%j', this.vars)
  }
}

let net = new nn6.Net(new Graph1())
net.run(300)
