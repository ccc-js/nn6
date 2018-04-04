const N = require('./node')
const G = require('./gate')
const tr = require('./matrix').transpose

class Layer extends G.CGate {
  constructor(inputs, out, Gate) {
    super(inputs, out)
    for (let i = 0; i < inputs.length; i++) {
      let g = new Gate(inputs[i], out[i])
      this.gates.push(g)
    }
  }
  output(v) {
    let size = this.out.length
    for (let i=0; i<size; i++) {
      this.out[i].value = v[i]
    }
  }
}

class SqrtError extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    let predicts = inputs[0]
    let facts = inputs[1]
    let len = predicts.length
    let e = net.newVector(len)
    let e2 = net.newVector(len)
    this.gates = [
      new Layer(tr([predicts, facts]), e, G.Sub),
      new Layer(tr([e, e]), e2, G.Mul),
      new G.Sum(e2, out)
    ]
    this.dumps = {predicts: predicts, facts: facts, e: e, e2, e2, out: out}
  }
}

class Net extends G.CGate {
  constructor() {
    super([], null)
    this.vars = {}
    this.varCount = 0
    this.dumpVars = {}
  }
  addVariable(value=0, name) {
    name = name || '$v' + this.varCount
    var v = this.vars[name]
    if (v == null) {
      v = new N.Variable(value)
      this.vars[name] = v
      this.varCount ++
    }
    return v
  }
  addVariables(list) {
    for (let name of list) {
      this.addVariable(0, name)
    }
    return this.vars
  }
  newConstants(list) {
    let a = []
    for (let name of list) {
      a.push(new N.Constant(name))
    }
    return a
  }
  newVector(n) {
    let v = []
    for (let i=0; i<n; i++) {
      v.push(this.addVariable(0))
    }
    return v
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
  adjust (step) {
    let nodes = Object.values(this.vars)
    for (let node of nodes) {
      node.adjust(step)
    }
  }
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
  Layer,
  SqrtError,
  Net,
}
