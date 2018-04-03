const N = require('./node')
const G = require('./gate')

function adjustNodes(nodes, step) {
  for (let node of nodes) {
    node.adjust(step)
  }
}

class Net extends G.IGate {
  constructor(inputs, out) {
    super(inputs, out)
    // this.inputs = inputs
    // this.out = out
    this.gates = []
  }
  forward () {
    let gates = this.gates, len = gates.length
    for (let i = 0; i < len; i++) {
      gates[i].forward()
    }
  }
  backward () {
    let gates = this.gates, len = gates.length
    for (let i = len - 1; i >= 0; i--) {
      gates[i].backward()
    }
  }
  adjust(step) {
    for (let g of this.gates) {
      g.adjust(step)
    }
    // adjustNodes(this.inputs)
    /*
    for (let node of this.inputs) {
      node.adjust(step)
    }
    */
  }
}

class Layer extends Net {
  constructor(inputs, out, Gate) {
    super(inputs, out)
    // this.inputs = inputs
    // this.out = out
    // this.gates = []
    let o = out
    let rows = o.length
    for (let j = 0; j < rows; j++) {
      let ginputs = getCol(inputs, j)
      let g = new Gate(ginputs, out[j])
      this.gates.push(g)
    }
  }
  output(v) {
    let size = this.out.length
    for (let i=0; i<size; i++) {
      this.out[i].value = v[i]
    }
  }


  /*
  forward () {
    let gsize = this.gates.length
    for (let i=0; i<gsize; i++) {
      this.gates[i].forward()
    }
  }
  backward () {
    let gsize = this.gates.length
    for (let i = gsize-1; i >= 0; i--) {
      this.gates[i].backward()
    }
  }
  */
}

class SqrtError extends Net {
  constructor(inputs, out) {
    super(inputs, out)
    let predicts = inputs[0]
    let facts = inputs[1]
    let len = predicts.length
    let e = N.newVector(len)
    let e2 = N.newVector(len)
    this.gates = [
      new Layer([predicts, facts], e, G.Sub),
      new Layer([e, e], e2, G.Mul),
      new G.Sum(e2, out)
    ]
    /*
    new nn6.net.Layer([[r1, r2], [c3, c2]], [e1, e2], nn6.gate.Sub),
    new nn6.net.Layer([[e1, e2], [e1, e2]], [e1s, e2s], nn6.gate.Mul),
    new nn6.gate.Sum([e1s, e2s], f),
    */
    this.dumps = {predicts: predicts, facts: facts, e: e, e2, e2, out: out}
  }
  forward () {
    super.forward()
    console.log('forward: %j', this.dumps)
  }
  backward () {
    super.backward()
    console.log('backward %j', this.dumps)
  }
}

function getCol(m, j) {
  let r = []
  let rows = m.length
  for (let i=0; i < rows; i++) {
    r.push(m[i][j])
  }
  return r
}

class Graph extends Net {
  constructor() {
    super([], null)
    // this.inputs = []
    // this.gates = []
    this.vars = {}
    this.dumpVars = {}
    // this.out = null
  }
  addVariables(list) {
    Object.assign(this.vars, N.newVariables(list))
    return this.vars
  }
  setDumpVariables(list) {
    this.dumpVars = {}
    for (let k of list) {
      this.dumpVars[k] = this.vars[k]
    }
  }
  addInputs(inputs) {
    this.inputs = this.inputs.concat(inputs)
  }
  addGates(gates) {
    this.gates = this.gates.concat(gates)
  }
  setOutput(out) {
    this.out = out
  }
  /*
  // 問題出在這裡， adjust 並沒有真正調整 sqrtError 的 value
  adjust (step) {
    // 3 = "error"
    let inputs = this.inputs
    for (let node of inputs) {
      if (node instanceof Array) {
        for (let n of node) {
          n.adjust(step)
        }
      } else {
        node.adjust(step)
      }
    }
  }
  */
  resetGrad () {
    let nodes = Object.values(this.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  gd(step, i) {
    this.forward()
    this.resetGrad()
    this.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  dump() {
    let r=[]
    for (let k in this.dumpVars) {
      r.push(k+':'+this.dumpVars[k].value.toFixed(6))
    }
    return r.join(' ')
  }
}

module.exports = {
  Net,
  Layer,
  SqrtError,
  Graph,
}
