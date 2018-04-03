const nn6 = require('../lib/nn6')

// 2x+y=3
//  x+y=2
// 解答： x=1, y=1
class G1 extends nn6.net.Graph {
  constructor () {
    super()
    let {x, y, r1, r2, r11, r12, r21, r22, e1, e2, e1s, e2s, f} = this.addVariables(['x', 'y', 'r1', 'r2', 'r11', 'r12', 'r21', 'r22', 'e1', 'e2', 'e1s', 'e2s', 'f'])
    this.setDumpVariables(['x', 'y', 'f'])
    let [c1, c2, c3] = nn6.node.newConstants([1, 2, 3])
    this.addInputs([[x, y], [c1, c2]])
    this.setOutput(f)
    this.addGates([
      new nn6.net.Layer([[x, y], [c2, c1]], [r11, r12], nn6.gate.Mul),
      new nn6.net.Layer([[x, y], [c1, c1]], [r21, r22], nn6.gate.Mul),
      new nn6.gate.Sum([r11, r12], r1),
      new nn6.gate.Sum([r21, r22], r2),
      
      new nn6.net.SqrtError([[r1, r2], [c3, c2]], f)
      
      /*
      new nn6.net.Layer([[r1, r2], [c3, c2]], [e1, e2], nn6.gate.Sub),
      new nn6.net.Layer([[e1, e2], [e1, e2]], [e1s, e2s], nn6.gate.Mul),
      new nn6.gate.Sum([e1s, e2s], f),
      */
    ])
    console.log('vars=%j', this.vars)
  }
}

nn6.GradientDescendent(new G1(), 100)
