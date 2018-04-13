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
    this.gates = [
      new Layer(tr([predicts, facts]), e, G.Sub),
      new VDot([e, e], out, net),
    ]
  }
}

class Neuron extends G.CGate {
  constructor(inputs, out, net, wMin, wMax, Sig=G.Sigmoid) {
    super(inputs, out)
    this.net = net
    let x = inputs
    let [n1] = N.newConstants([-1])
    let ex = [n1].concat(x)
    let w = net.newVector(ex.length, wMin, wMax)
    let s = net.addVariable(0, null, 'neuron.s')
    let g1 = new VDot([ex, w], s, net)   // ex*w = s
    let g2 = new Sig([s], out) // G.Sigmoid([s], out)  // sigmoid(s) = out
    // let g2 = new G.Relu([s], out)        // relu(s) = out
    // let g2 = new G.Tanh([s], out)     // tanh(s) = out
    this.gates.push(g1)
    this.gates.push(g2)
  }
}

class ConnectLayer extends G.CGate {
  constructor(inputs, out, net, wMin, wMax, Sig=G.Sigmoid) {
    super(inputs, out)
    this.net = net
    for (let i=0; i<out.length; i++) {
      this.gates.push(new Neuron(inputs, out[i], net, wMin, wMax, Sig))
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

// Rnn 測試成功了 ....
// 參考： https://github.com/karpathy/recurrentjs/blob/master/src/recurrent.js
// var h0 = G.mul(model['Wxh'+d], input_vector);
// var h1 = G.mul(model['Whh'+d], hidden_prev);
// var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));
class Rnn extends G.CGate {
  constructor(inputs, hSize, out, net) {
    super(inputs, out)
    this.net = net
    let hi = net.newVector(hSize)
    let hh = net.newVector(hSize)
    let hidden = net.newVector(hSize)
    let s = net.newVector(out.length)
    this.gates = [
      new ConnectLayer(inputs, hi, net, -0.2, 0.2), // , G.Tanh
      new ConnectLayer(hidden, hh, net, -0.2, 0.2), // 循環層, , G.Tanh
      new Layer([hi, hh], hidden, G.Add),
      // new Layer([hidden], G.Relu),
      new ConnectLayer(hidden, out, net, -1.0, 1.0)
    ]
  }
}

/* 參考: https://github.com/karpathy/recurrentjs/blob/master/src/recurrent.js#L254
  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(var i=0,n=m.w.length;i<n;i++) { 
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
    }
*/

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

class Rnn01 extends Net {
  constructor () {
    super()
    let {x, f} = this.addVariables(['x', 'w1', 'w2', 'f'])
    this.setDumpVariables(['x', 'f'])
    this.inputs = [ x ]
    this.out = [ f ]
    this.moment = 0.01
    this.rate = 0.1
    this.gates = [
      new Rnn(this.inputs, 2, this.out, this)
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
  Rnn,
  Net,
  Perceptron2,
  Mlp2,
  Rnn01,
}
