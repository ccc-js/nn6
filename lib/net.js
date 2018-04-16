const N = require('./node')
const G = require('./gate')
const U = require('./util')
const tr = require('./matrix').transpose

class Layer extends G.CGate {
  constructor(inputs, out, Gate) {
    let I = tr(inputs)
    super(I, out)
    for (let i = 0; i < I.length; i++) {
      let g = new Gate(I[i], out[i])
      this.gates.push(g)
    }
  }
  output(v) {
    N.setValues(this.out, v)
    /*
    let size = this.out.length
    for (let i=0; i<size; i++) {
      this.out[i].value = v[i]
    }
    */
  }
}

class NetGate extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    if (net == null) throw Error('NetGate: net == null')
    this.net = net
  }
}

class VDot extends NetGate {
  constructor(inputs, out, net) {
    super(inputs, out, net)
    let a = inputs[0]
    let b = inputs[1]
    let ab = net.newVector(a.length)
    this.gates = [
      new Layer([a, b], ab, G.Mul),
      new G.Sum(ab, out)
    ]
  }
}

class MDot extends NetGate {
  constructor(inputs, out, net) {
    super(inputs, out, net)
    let m = inputs[0]
    let a = inputs[1]
    let mLen = m.length
    this.gates = []
    for (let i=0; i<mLen; i++) {
      this.gates.push(new VDot([m[i], a], out[i], net))
    }
  }
}

class SqrtError extends NetGate {
  constructor(inputs, out, net) {
    super(inputs, out, net)
    let predicts = inputs[0]
    let facts = inputs[1]
    let len = predicts.length
    let e = net.newVector(len)
    this.gates = [
      new Layer([predicts, facts], e, G.Sub),
      new VDot([e, e], out, net),
    ]
  }
}

class Neuron extends NetGate {
  constructor(inputs, out, net, wMin, wMax, Sig=G.Sigmoid) {
    super(inputs, out, net)
    let x = inputs[0]
    let [n1] = N.newConstants([-1])
    let b = inputs[1] || n1
    let ex = [b].concat(x)
    let w = net.newVector(ex.length, wMin, wMax)
    if (Sig == null) {
      this.gates.push(new VDot([ex, w], out, net))   // ex*w = out
    } else {
      let s = net.addVariable(0, null, 'neuron.s')
      this.gates.push(new VDot([ex, w], s, net))     // ex*w = s
      this.gates.push(new Sig([s], out))             // G.Sigmoid([s], out)
    }
  }
}

class Connect extends NetGate {
  constructor(inputs, out, net, wMin, wMax, Sig=G.Sigmoid) {
    super(inputs, out, net)
    let x = inputs[0]
    for (let i=0; i<out.length; i++) {
      this.gates.push(new Neuron([x], out[i], net, wMin, wMax, Sig))
    }
  }
}

class Net extends G.CGate {
  constructor() {
    super([], null)
    this.vars = {}
    this.varCount = 0
    this.dumpVars = {}
    this.step = 0.1
    this.moment = 0.01
  }
  addVariable(value=0, name, tag) {
    tag = tag || 't'
    // name = name || '$v' + this.varCount
    name = name || '$' + tag + ':' + this.varCount
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
  newVector(n, min=-1, max=1) {
    let v = []
    for (let i=0; i<n; i++) {
      v.push(this.addVariable(U.rand(min,max)))
    }
    return v
  }
  setDumpVariables(list) {
    this.dumpVars = {}
    for (let k of list) {
      this.dumpVars[k] = this.vars[k]
    }
  }
  adjust (step, moment) {
    let nodes = Object.values(this.vars)
    for (let node of nodes) {
      node.adjust(step, moment)
    }
  }
  resetGrad () {
    let nodes = Object.values(this.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  learn(input, fact) {
    N.setValues(this.inputs, input)
    // for (let i=0; i < input.length; i++) this.inputs[i].value = input[i]
    this.forward()
    this.resetGrad()
    let squareError = 0
    for (let oi=0; oi < fact.length; oi++) {
      let diff = fact[oi] - this.out[oi].value
      this.out[oi].grad = diff
      squareError += diff * diff
    }
    this.backward()
    this.adjust(this.step, this.moment)
    return squareError
  }
  gd() {
    this.forward()
    this.resetGrad()
    this.out.grad = 1.0
    this.backward()
    this.adjust(-1*this.step, this.moment)
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
  VDot,
  MDot,
  SqrtError,
  Neuron,
  Connect,
  NetGate,
  Net,
}
