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

class VDot extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    let a = inputs[0]
    let b = inputs[1]
    let len = a.length
    let ab = net.newVector(len)
    this.gates = [
      new Layer(tr([a, b]), ab, G.Mul),
      new G.Sum(ab, out)
    ]
  }
}

class MDot extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    let m = inputs[0]
    let a = inputs[1]
    let mLen = m.length
    this.gates = []
    for (let i=0; i<mLen; i++) {
      this.gates.push(new VDot([m[i], a], out[i], net))
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
  learn(input, fact, step) {
    for (let i=0; i < input.length; i++) this.inputs[0][i].value = input[i]
    this.forward()
    this.resetGrad()
    let diffSum = 0
    for (let oi=0; oi < fact.length; oi++) {
      let diff = fact[oi] - this.out[oi].value
      diffSum += diff
      this.out[oi].grad = this.out[oi].value - fact[oi]
    }
    this.backward()
    this.adjust(step)
    return diffSum
  }
  gd(step) {
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
/*
class Learner extends Net {
  constructor(inputs, facts) {
    this.inputs = inputs
    this.facts = facts
  }
  learnSample(input, out) {
    this.
    this.gd(step)
  }
  backward() {
    let iLen = this.inputs.length
    for (let i=0; i<iLen; i++) {
      learnSample(net, this.inputs[i], this.facts[i])
      net.forward()
      console.log(i + ':' + this.net.dump())
    }
  }
  gd(step, i) {
    let iLen = this.inputs.length
    for (let i=0; i<iLen; i++) {
      learnSample(net, this.inputs[i], this.facts [i])
      net.forward()
      console.log(i + ':' + this.net.dump())
    }
  this.forward()
    this.resetGrad()
    this.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  learn(maxLoops) {
    for (let loop = 0; loop < maxLoops; loop++) {
    }
  }
}
*/

module.exports = {
  Layer,
  VDot,
  MDot,
  SqrtError,
  Net,
}
