const N = require('./node')
const G = require('./gate')
const tr = require('./matrix').transpose

function rand(a,b) {
  return a + Math.random() * (b-a)
}

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
    let size = this.out.length
    for (let i=0; i<size; i++) {
      this.out[i].value = v[i]
    }
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

class Mlp extends NetGate {
  constructor(inputs, hSize, out, net) {
    super(inputs, out, net)
    let hidden = net.newVector(hSize)
    this.gates = [
      new Connect([inputs], hidden, net, -0.2, 0.2),
      new Connect([hidden], out, net, -2.0, 2.0)
    ]
  }
}


class XhConnect extends NetGate {
  constructor(inputs, out, net, wMin, wMax, Sig=G.Sigmoid) {
    super(inputs, out, net)
    let x = inputs[0]
    let h = inputs[1]
    let hSize = h.length
    let b  = (inputs[2] == null) ? net.newVector(hSize) : inputs[2]
    let wx = net.newVector(hSize)
    let uh = net.newVector(hSize)
    let f  = net.newVector(hSize)
    this.gates = [
      new Connect([x], wx, net, wMin, wMax, null),
      new Connect([h], uh, net, wMin, wMax, null),
      new Layer([wx, uh, b], f, G.Sum),
      new Layer([f], out, Sig),
    ]
  }
}

class Rnn extends NetGate {
  activate(input) {
    for (let i=0; i<input.length; i++) this.inputs[i].value = input[i]
    this.forward()
    return this.out
  }
}

// Rnn 測試成功了 ....
// 參考： https://github.com/karpathy/recurrentjs/blob/master/src/recurrent.js
// var h0 = G.mul(model['Wxh'+d], input_vector);
// var h1 = G.mul(model['Whh'+d], hidden_prev);
// var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));
// 問題是、karpathy 加完後才用 relu，我是直接在 Connect 中用 sigmoid。

class SimpleRnn extends Rnn {
  constructor(inputs, hSize, out, net) {
    super(inputs, out, net)
    let hidden = net.newVector(hSize)
    this.gates = [
      new XhConnect([inputs, hidden], hidden, net, -0.2, 0.2), // 循環層
      new Connect([hidden], out, net, -1.0, 1.0)
    ]
  }
}

// 參考: https://en.wikipedia.org/wiki/Long_short-term_memory
// 其中的 LSTM with forget gate
class Lstm extends Rnn {
  constructor(inputs, out, net) {
    super(inputs, out, net)
    let x = inputs[0]
    let iSize = x.length
    let hSize = out.length
    let f = net.newVector(hSize)
    let i = net.newVector(hSize)
    let c = net.newVector(hSize)
    let ci = net.newVector(hSize)
    let ic = net.newVector(hSize)
    let fc = net.newVector(hSize)
    let h = net.newVector(hSize)
    let hi = net.newVector(hSize)
    this.gates = [
      new XhConnect([x, h], f, net, -0.2, 0.2),
      new XhConnect([x, h], i, net, -0.2, 0.2),
      new XhConnect([x, h], out, net, -1.0, 1.0),
      new XhConnect([x, h], ci, net, -0.2, 0.2, G.tanh),
      new Layer([i, ci], ic, G.Mul),
      new Layer([f, c], fc, G.Mul),
      new Layer([ic, fc], c, G.Add),
      new Layer([c], hi, G.Sigmoid),
      new Layer([out, hi], h, G.Mul),
    ]
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
    for (let i=0; i < input.length; i++) this.inputs[i].value = input[i]
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

class Perceptron2 extends Net {
  constructor () {
    super()
    let {x1, x2, f} = this.addVariables(['x1', 'x2', 'f'])
    this.setDumpVariables(['x1', 'x2', 'f'])
    let x = [x1, x2]
    this.inputs = [ x ]
    this.out = [ f ]
    this.gates = [
      new Neuron([x], f, this)
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

class RnnNet extends Net {
  activate(input) {
    let out = this.gates[0].activate(input)
    let o = []
    for (let i=0; i<out.length; i++) {
      let ov = out[i].value
      o.push((ov > 0.5) ? 1 : 0)
    }
    return o
  }
  generate(input, len) {
    let o = input
    for (let i=0; i<len; i++) {
      o = this.activate(o)
      console.log('o=%j', o)
    }
  }
}

class Rnn01 extends RnnNet {
  constructor () {
    super()
    let {x, f} = this.addVariables(['x', 'f'])
    this.setDumpVariables(['x', 'f'])
    this.inputs = [ x ]
    this.out = [ f ]
    // this.moment = 0.01
    // this.rate = 0.1
    this.gates = [
      new SimpleRnn(this.inputs, 2, this.out, this)
    ]
  }
}

class Lstm01 extends RnnNet {
  constructor () {
    super()
    let {x, f} = this.addVariables(['x', 'f'])
    this.setDumpVariables(['x', 'f'])
    this.inputs = [ x ]
    this.out = [ f ]
    // this.moment = 0.01
    // this.rate = 0.1
    this.gates = [
      new Lstm(this.inputs, this.out, this)
    ]
  }
}

module.exports = {
  Layer,
  VDot,
  MDot,
  SqrtError,
  Neuron,
  Connect,
  Mlp,
  Rnn,
  Net,
  Perceptron2,
  Mlp2,
  Rnn01,
  Lstm01
}
