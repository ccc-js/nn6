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

class Neuron extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    let x = inputs
    let [n1] = N.newConstants([-1])
    let ex = [n1].concat(x)
    let w = net.newVector(ex.length)
    let s = net.addVariable(0, null, 'neuron.s')
    let g1 = new VDot([ex, w], s, net)   // ex*w = s
    let g2 = new G.Sigmoid([s], out)     // sigmoid(s) = out
    // let g2 = new G.Relu([s], out)
    // let g2 = new G.Tanh([s], out)     // tanh(s) = out
    this.gates.push(g1)
    this.gates.push(g2)
  }
}

class SigLayer extends G.CGate {
  constructor(inputs, out, net) {
    super(inputs, out)
    this.net = net
    for (let i=0; i<out.length; i++) {
      this.gates.push(new Neuron(inputs, out[i], net))
    }
  }
}

class Mlp extends G.CGate {
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

class Net extends G.CGate {
  constructor() {
    super([], null)
    this.vars = {}
    this.varCount = 0
    this.dumpVars = {}
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
  learnAll(inputs, facts, step) {
    let diffs = new Array(facts.length)
    diffs.fill(0)
    let energy = 0
    for (let si = 0; si < inputs.length; si++) {
      let squareError = 0
      for (let i=0; i < input.length; i++) {
        this.inputs[i].value = input[i]
      }
      this.forward()
      for (let i=0; i < fact.length; i++) {
        let diff = this.out[i].value - fact[i]
        diffs[i] += diff
        // this.out[i].grad = diff
        squareError += diff * diff
      }
      energy += squareError
    }
    console.log('  ' + i + ' => ' + net.dump())
    this.backward()
    // console.log('  this.inputs=%j', this.inputs)
    // console.log('  this.vars=%j', this.vars)
    // console.log('  this.out=%j', this.out)
    this.adjust(step)
    return squareError
  }
  /*
  learn(input, fact, step) {
    for (let i=0; i < input.length; i++) this.inputs[i].value = input[i]
    this.forward()
    // console.log('  input=%j', input)
    this.resetGrad()
    let squareError = 0
    for (let oi=0; oi < fact.length; oi++) {
      let diff = this.out[oi].value - fact[oi]
      this.out[oi].grad = diff
      squareError += diff * diff
    }
    this.backward()
    // console.log('  this.inputs=%j', this.inputs)
    // console.log('  this.vars=%j', this.vars)
    // console.log('  this.out=%j', this.out)
    this.adjust(step)
    return squareError
  }
  */
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

class Perceptron2 extends Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    this.inputs = [ x1, x2 ]
    this.out = [ f ]
    this.gates = [
      new Neuron(this.inputs, f, this)
    ]
  }
}

class Mlp2 extends Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    this.inputs = [ x1, x2 ]
    this.out = [ f ]
    this.gates = [
      new Mlp(this.inputs, 3, this.out, this)
    ]
  }
}

module.exports = {
  Layer,
  VDot,
  MDot,
  SqrtError,
  Neuron,
  SigLayer,
  Mlp,
  Net,
  Perceptron2,
  Mlp2
}
