const N = require('./node')
const G = require('./gate')
const U = require('./util')
const L = require('./loss')
const F = require('./func')
const tr = require('./matrix').transpose

/*

https://github.com/karpathy/convnetjs/blob/master/src/convnet_net.js

檢查目前還沒實作的 layer 有哪些？

switch(def.type) {
  // case 'fc': this.layers.push(new global.FullyConnLayer(def)); break; 
  case 'lrn': this.layers.push(new global.LocalResponseNormalizationLayer(def)); break;
  case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
  // case 'input': this.layers.push(new global.InputLayer(def)); break;
  // case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
  * case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
  * case 'conv': this.layers.push(new global.ConvLayer(def)); break; // https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_dotproducts.js
  * case 'pool': this.layers.push(new global.PoolLayer(def)); break; // https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_pool.js
  // case 'relu': this.layers.push(new global.ReluLayer(def)); break;
  // case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
  // case 'tanh': this.layers.push(new global.TanhLayer(def)); break;
  case 'maxout': this.layers.push(new global.MaxoutLayer(def)); break;
  * case 'svm': this.layers.push(new global.SVMLayer(def)); break;
  default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
}

Conv : https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_dotproducts.js
Regression + Svm + Softmax : https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_loss.js
Pool : https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_pool.js

Relu, Sigmoid, Tanh, MaxoutLayer : https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_nonlinearities.js
*/

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
  }
}

// 注意：此處的 SoftmaxLayer 是個 1 to 1 的 Layer
// 這種情況下的梯度很簡單， dx = o * (1-o) * df
// 如果是全連接層，就要考慮 i != j 的傾斜連接情況 (但對目前 1 to 1 的情況不用)
// Softmax 的梯度請參考： https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
class SoftmaxLayer extends G.Gate {
  forward () {
    let x = N.getValues(this.inputs)
    let o = F.softmax(x)
    N.setValues(this.out, o)
  }
  backward () {
    let o = this.out 
    for (let i=0; i<this.inputs.length; i++) {
      this.inputs[i].grad = o[i].value * (1 - o[i].value) * o[i].grad
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

class SquareError extends NetGate {
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
    this.dumpList = []
    this.step = 0.1
    this.moment = 0.01
    this.fLoss = L.squareError
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
    this.dumpList = list
  }
  adjust (step = this.step, moment = this.moment) {
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
  learn (input, fact) {
    N.setValues(this.inputs, input)
    this.forward()
    this.resetGrad()
    let o = N.getValues(this.out)
    for (let oi=0; oi < fact.length; oi++) {
      let diff = fact[oi] - this.out[oi].value
      this.out[oi].grad = diff
    }
    this.backward()
    this.adjust(this.step, this.moment)
    return this.fLoss(fact, o)
  }
  gd() {
    this.forward()
    this.resetGrad()
    this.out.grad = 1.0
    this.backward()
    this.adjust(-1*this.step, this.moment)
  }
  list(vars) {
    let r=[]
    for (let k of vars) {
      let value = this.vars[k].value
      if (Math.abs(value) < 0.0001) value = 0
      r.push(k+':'+value.toFixed(4))
    }
    return r.join(' ')
  }    
  dump() {
    return this.list(this.dumpList)
  }
}


module.exports = {
  Layer,
  SoftmaxLayer,
  VDot,
  MDot,
  SquareError,
  Neuron,
  Connect,
  NetGate,
  Net,
}
