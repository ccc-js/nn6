const N = require('./node')
const G = require('./gate')
const tr = require('./matrix').transpose

function rand(a,b) {
  return a + Math.random() * (b-a)
}

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
    let ab = net.newVector(a.length)
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
    // let e2 = net.addVariable(0)
    this.gates = [
      new Layer(tr([predicts, facts]), e, G.Sub),
      new VDot([e, e], out, net),
      // new VDot([e, e], e2, net),
      // new G.Pow([e2], 0.5, out)
      /*
      new Layer(tr([e, e]), e2, G.Mul),
      new G.Sum(e2, out)
      */
    ]
  }
}

class Neuron extends G.CGate {
  constructor(inputs, out, net, wMin, wMax) {
    super(inputs, out)
    this.net = net
    let x = inputs
    let [n1] = N.newConstants([-1])
    let ex = [n1].concat(x)
    let w = net.newVector(ex.length, wMin, wMax)
    let s = net.addVariable(0, null, 'neuron.s')
    let g1 = new VDot([ex, w], s, net)   // ex*w = s
    let g2 = new G.Sigmoid([s], out)  // sigmoid(s) = out
    // let g2 = new G.Relu([s], out)        // relu(s) = out
    // let g2 = new G.Tanh([s], out)     // tanh(s) = out
    this.gates.push(g1)
    this.gates.push(g2)
  }
}

class ConnectLayer extends G.CGate {
  constructor(inputs, out, net, wMin, wMax) {
    super(inputs, out)
    this.net = net
    for (let i=0; i<out.length; i++) {
      this.gates.push(new Neuron(inputs, out[i], net, wMin, wMax))
    }
  }
}

class Mlp extends G.CGate {
  constructor(inputs, hSize, out, net) {
    super(inputs, out)
    this.net = net
    let hidden = net.newVector(hSize)
    this.gates = [
      new ConnectLayer(inputs, hidden, net, -0.2, 0.2),
      new ConnectLayer(hidden, out, net, -2.0, 2.0)
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
  newVector(n, min=-1, max=1) {
    let v = []
    for (let i=0; i<n; i++) {
      v.push(this.addVariable(rand(min,max)))
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
  /*
  learnAll(loop, inputs, facts, step) {
    console.log('loop=%j inputs=%j facts=%j step=%j', loop, inputs, facts, step)
    let diffs = new Array(this.out.length)
    diffs.fill(0)
    let energy = 0
    for (let si = 0; si < inputs.length; si++) {
      let input = inputs[si], fact = facts[si]
      for (let i=0; i < input.length; i++) {
        this.inputs[i].value = input[i]
      }
      this.forward()
      console.log('  ' + si + ' => ' + this.dump())
      let squareError = 0
      // console.log('fact=%j', fact)
      for (let i=0; i < fact.length; i++) {
        let diff = this.out[i].value - fact[i]
        diffs[i] = diff
        // this.out[i].grad = diff
        squareError += diff * diff
      }
      console.log('   diffs=%j', diffs)
      this.resetGrad()
      for (let i=0; i<diffs.length; i++) {
        this.out[i].grad = diffs[i]
      }
      this.backward()
      this.adjust(step)
      energy += squareError
    }
    console.log('%d:energy = %d', loop, energy)
    return energy
  }
  */
  learn(input, fact, step) {
    for (let i=0; i < input.length; i++) this.inputs[i].value = input[i]
    this.forward()
    // console.log('  input=%j', input)
    this.resetGrad()
    let squareError = 0
    for (let oi=0; oi < fact.length; oi++) {
      let diff = fact[oi] - this.out[oi].value
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
  ConnectLayer,
  Mlp,
  Net,
  Perceptron2,
  Mlp2
}
