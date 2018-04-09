const nn6 = require('../lib/nn6')

class Perceptron extends nn6.gate.CGate {
  constructor(inputs, out, energy, net) {
    super(inputs, out)
    this.net = net
    let diff = net.newVector(inputs.length)
    let f = net.newVector(inputs.length)
    let s = net.newVector(inputs.length)
    let e = net.newVector(inputs.length)
    let e2 = net.newVector(inputs.length)
    for (let i=0; i<inputs.length; i++) {
      let w = inputs[i][0]
      let x = inputs[i][1]
      let g1 = new nn6.net.VDot([w, x], f[i], net)      // x*w = f
      let g2 = new nn6.gate.Sigmoid([f[i]], s[i])       // sigmoid(f) = s
      let g3 = new nn6.gate.Sub([s[i], out[i]], e[i])   // s-out = e
      let g4 = new nn6.gate.Mul([e[i], e[i]], e2[i])    // e * e = e2
      this.gates.push(g1)
      this.gates.push(g2)
      this.gates.push(g3)
      this.gates.push(g4)
    }
    this.gates.push(new nn6.gate.Sum(e2, energy, net))  // sum(e2) = energy
  }
  forward () {
    super.forward()
    // console.log('')
  }
}

// andTable = [ [ 0, 0, 0 ], [ 0, 1, 0 ], [ 1, 0, 0 ], [ 1, 1, 1 ] ]
// [w0, w1, w2] * [x0, x1, x2] = s, sig(s) = f
// [0,   1,  1] * [-1, x1, x2] = 1, sig(s) = f
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {w0, w1, w2, energy} = this.addVariables(['w0', 'w1', 'w2', 'energy'])
    this.setDumpVariables(['w0', 'w1', 'w2', 'energy'])
    let [n1, c0, c1] = nn6.node.newConstants([-1, 0, 1])
    this.gates = [ new Perceptron([
      [[w0, w1, w2], [n1, c0, c0]],
      [[w0, w1, w2], [n1, c0, c1]],
      [[w0, w1, w2], [n1, c1, c0]],
      [[w0, w1, w2], [n1, c1, c1]]],
      [c0, c0, c0, c1], energy, this)
    ]
    this.out = energy
  }
}

nn6.GradientDescendent(new Net1(), 1000)

