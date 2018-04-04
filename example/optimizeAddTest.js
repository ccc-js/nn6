const nn6 = require('../lib/nn6')

// f(x,y) = x^2 + 2xy + y^2 + 2
// 在 x=0 y=0 時，有最小值 2
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let v = this.addVariables(['x', 'y', 'xx', 'yy', 'f'])
    this.setDumpVariables(['x', 'y', 'f'])
    let [c2] = this.newConstants([2])
    this.addInputs([v.x, v.y])
    v.x.value = 1
    v.y.value = 2
    this.setOutput(v.f)
    this.addGates([
      new nn6.gate.Mul([v.x, v.x], v.xx),
      new nn6.gate.Mul([v.y, v.y], v.yy),
      new nn6.gate.Add([v.xx, v.yy], v.f)
    ])
  }
}

nn6.GradientDescendent(new Net1(), 300)
