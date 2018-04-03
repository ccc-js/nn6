const gate = require('./gate')
const node = require('./node')

class Layer extends gate.Gate {
  constructor(inputs, out, Gate) {
    this.inputs = inputs
    this.out = out
    let o = out, olen = o.length, isize=inputs.len, ginputs = []
    for (let oi=0; oi< olen; oi++) {
      for (let i=0; i < isize; i++) {
        ginputs.push(inputs[oi][i])
      }
      let g = new Gate(ginputs, out[oi])
      this.gates.push(g)
    }
  }
  output(v) {
    let size = this.out.length
    for (let i=0; i<size; i++) {
      this.out[i].value = v[i]
    }
  }
  forward () {
    let gsize = this.gates.length
    for (let i=0; i<gsize; i++) {
      this.gates[i].forward()
    }
    return this.out
  }
  backward () {
    let gsize = this.gates.length
    for (let i=size-1; i>=0; i--) {
      this.gates[i].backward()
    }
  }
}

class Neuron extends Layer {
  constructor(inputs, out) {
    super(inputs, out)
    let x = [node.C1].concat(inputs[0])
    let w = [node.C0].concat(inputs[1])
    let mulLayer = new Layer([x, w], xw, gate.Mul)
    let sumLayer = new Layer([xw], out, gate.Sum)
  }
}
