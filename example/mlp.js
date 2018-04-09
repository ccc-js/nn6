const nn6 = require('../lib/nn6')

class SigLayer extends  nn6.gate.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    let x = inputs[0]
    let f = net.newVector(out.length)
    for (let i=0; i<out.length; i++) {
      let w = net.newVector(x.length)
      let g1 = new nn6.net.VDot([w, x], f[i], net)      // x*w = f
      let g2 = new nn6.gate.Sigmoid([f[i]], out[i])     // sigmoid(f) = s
      this.gates.push(g1)
      this.gates.push(g2)
    }
  }
}

class Mlp extends nn6.gate.CGate {
  constructor(inputs, hSize, out, net) {
    super(inputs, out)
    this.net = net
    let hidden = net.newVector(hSize)
    this.gates = [
      new SigLayer(inputs, hidden, net),
      new SigLayer(hidden, out, net)
    ]
  }
}

class MlpPredictor extends nn6.gate.CGate {
  constructor(inputs, hSize, ans, energy, net) {
    super(inputs, ans)
    this.net = net
    let hidden = net.newVector(hSize)
    let out = net.newVector(ans[0].length)
    3=5
    this.gates = [ new Mlp(x, hSize, out, net) ] // x 如何接上 input ???
    let e = net.newVector(inputs.length)
    let e2 = net.newVector(inputs.length)
    for (let i=0; i<inputs.length; i++) {
      let g3 = new nn6.gate.Sub([out[i], ans[i]], e[i])   // s - out = e
      let g4 = new nn6.gate.Mul([e[i], e[i]], e2[i])    // e * e = e2
      this.gates.push(g3)
      this.gates.push(g4)
    }
    this.gates.push(new nn6.gate.Sum(e2, energy, net))  // sum(e2) = energy
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
    this.gates = [ new MlpPredictor([
      [[w0, w1, w2], [n1, c0, c0]],
      [[w0, w1, w2], [n1, c0, c1]],
      [[w0, w1, w2], [n1, c1, c0]],
      [[w0, w1, w2], [n1, c1, c1]]],
      2, [c0, c1, c1, c0], energy, this)
    ]
    this.out = energy
  }
}

nn6.GradientDescendent(new Net1(), 10)

