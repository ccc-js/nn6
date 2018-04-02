var D = require('./diff')
var j6 = require('j6')

class Constant {
  constructor(value) {
    this.val = value // 前向函數執行結果 Gate Output 為常數
    this.grad = 0   // 反向傳播梯度 Gradient = 0
  }
  get value() { return this.val }
  set value(v) { console.log('Constant cannot be set value') }
  get grad() { return 0 }
  set grad(v) { }
}

class Variable {
  constructor(value, grad) {
    this.value = value // 前向函數執行結果 Gate Output
    this.grad = grad || 0.0 // 反向傳播梯度 Gradient
  }
}

class Gate {
  constructor(inputs, out, f, dfx, dfy) {
    this.inputs = inputs
    this.out = out
    this.f = f
    this.dfx = dfx
    this.dfy = dfy
  }
  output(v) {
    this.out.value = v
    return this.out
  }

  get x() { return this.inputs[0].value }
  get y() { return this.inputs[1].value }

  forward () {
    let o = this.f(this.x, this.y)
    return this.output(o)
  }

  backward () {
    this.inputs[0].grad += this.dfx(this.x, this.y) * this.out.grad
    // 注意：以下預設的方法，只在對稱的情況下可以使用，例如 x+y, x*y, 在不對稱的情況不能使用，例如: x-y, x^y. 
    if (this.dfy != null) {
      this.inputs[1].grad += this.dfy(this.x, this.y) * this.out.grad
    } else if (this.inputs[1] != null) {
      this.inputs[1].grad += this.dfx(this.y, this.x) * this.out.grad
    }
  }
}

class Neg extends Gate {
  constructor(inputs, out) { super(inputs, out, D.neg, D.dneg) }
}

class Rev extends Gate {
  constructor(inputs, out) { super(inputs, out, D.rev, D.drev) }
}

class Relu extends Gate {
  constructor(inputs, out) { super(inputs, out, D.relu, D.drelu) }
}

class Sigmoid extends Gate {
  constructor(inputs, out) { super(inputs, out, D.sigmoid, D.dsigmoid) }
}

// Binary Operation
class Add extends Gate {
  constructor(inputs, out) { super(inputs, out, D.add, D.dadd) }
}

class Sub extends Gate {
  constructor(inputs, out) { super(inputs, out, D.sub, D.dsub, D.dsuby) }
}

class Mul extends Gate {
  constructor(inputs, out) { super(inputs, out, D.mul, D.dmul) }
}

class Div extends Gate {
  constructor(inputs, out) { super(inputs, out, D.div, D.ddiv, D.ddivy) }
}

class Exp extends Gate {
  constructor(inputs, out) { super(inputs, out, D.exp, D.dexp) }
}

class Pow extends Gate {
  constructor(inputs, out) { super(inputs, out, D.pow, D.dpow, D.dpowy) }
}

class Sum extends Gate {
  forward () {
    let s = 0
    for (let x of this.inputs) s += x.value
    return this.output(s)
  }
  backward () {
    for (let x of this.inputs) {
      x.grad +=  1 * this.out.grad
    }
  }
}

class Times extends Gate {
  forward () {
    let s = 1
    for (let x of this.inputs) s *= x.value
    return this.output(s)
  }
  backward () {
    let s = 1
    for (let x of this.inputs) s *= x.value
    for (let x of this.inputs) {
      x.grad += (s / x.value) * this.out.grad
    }
  }
}

class Graph {
  constructor() {
    this.inputs = []
    this.gates = []
    this.vars = {}
    this.out = null
  }
  static newVariables(list) {
    let d = {}
    for (let v of list) {
      d[v] = new Variable(0)
    }
    return d
  }
  static newConstants(list) {
    let a = []
    for (let v of list) {
      a.push(new Constant(v))
    }
    return a
  }
  addVariables(list) {
    Object.assign(this.vars, Graph.newVariables(list))
    return this.vars
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
}

class Net {
  constructor(graph) {
    this.graph = graph
  }
  forward () {
    let gates = this.graph.gates
    for (let i = 0; i < gates.length; i++) {
      gates[i].forward()
    }
  }
  backward () {
    let gates = this.graph.gates
    for (let i = gates.length - 1; i >=0; i--) {
      gates[i].backward()
    }
  }
  adjust (step) {
    let inputs = this.graph.inputs
    for (let node of inputs) {
      node.value += step * node.grad
    }
  }
  resetGradient() {
    let nodes = Object.values(this.graph.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  gd(step) {
    this.forward()
    this.resetGradient()
    this.graph.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  run(maxLoops) {
    let lastValue = 9999
    let out = this.graph.out
    for (let i = 0; i < maxLoops; i++) {
      this.gd(-0.01)
      // console.log('  inputs:%j', this.inputs)
      console.log('out=%d', i, out.value)
      if (out.value > lastValue) break
      lastValue = out.value
    }
  }
}

module.exports = {
  Constant,
  Variable,
  // Gates
  Gate,
  Neg,
  Rev,
  Relu,
  Sigmoid,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Exp,
  Sum,
  Times,
  // Net
  Graph,
  Net
}

/*
// Uniary Operation
class Neg extends Gate {
  forward () { return this.output(-1 * this.inputs[0].value) }
  backward () { this.inputs[0].grad += -1 * this.out.grad }
}

class Exp extends Gate {
  forward () { return this.output(Math.exp(this.inputs[0].value)) }
  backward () { this.inputs[0].grad += this.out.grad * this.out.value }
}

class Relu extends Gate {
  forward () { return this.output(Math.exp(this.inputs[0].value)) }
  backward () { this.x.grad += this.out.value > 0 ? this.out.grad : 0.0 }
}

class Sigmoid extends Gate {
  forward () {
    // let x = this.inputs[0]
    return this.output(D.sigmoid(this.x.value))
  }
  backward () {
    // let x = this.inputs[0]
    var s = F.sigmoid(this.x.value); 
    this.x.grad += (s * (1 - s)) * this.out.grad
  }
}

// Binary Operation
class Pow extends Gate {
  forward () {
    return this.output(Math.pow(this.x.value, this.y.value))
  }
  backward () {
    let x = this.x, vx = x.value
    let y = this.y, vy = y.value
    x.grad += vy * Math.pow(vx, vy - 1) * this.out.grad
    y.grad += vx * Math.pow(vy, vx - 1) * this.out.grad
  }
}
*/
/*
class Mul {
  forward (x, y) {
    this.x = x
    this.y = y
    this.out = new Variable(x.value * y.value)
    return this.out
  }
  backward () {
    this.x.grad += this.x.value * this.out.grad
    this.y.grad += this.y.value * this.out.grad
  }
}

class Add {
  forward (x, y) {
    this.x = x
    this.y = y
    this.out = new Variable(x.value + y.value)
    return this.out
  }
  backward () {
    this.x.grad += 1 * this.out.grad
    this.y.grad += 1 * this.out.grad
  }
}

class Gate {
  constructor(inputs, internals, outputs) {
    this.inputs = []
    for (let i in this.inputs) {
      new Variable(x)
      this.vars[x] = 
      this.inputs[i] = 
    }
  }
  output(value) {
    this.out = new Variable(value)
    return this.out
  }
}
*/
