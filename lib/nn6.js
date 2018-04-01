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
  constructor(inputs, out) {
    this.inputs = inputs
    this.out = out
  }
  output(v) {
    this.out.value = v
    return this.out
  }
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

class Sigmoid extends Gate {
  static f(x) {
    return 1 / (1 + Math.exp(-x))
  }
  forward () {
    let x = this.inputs[0]
    return this.output(Sigmoid.f(x.value))
  }
  backward () {
    let x = this.inputs[0]
    var s = Sigmoid.f(x.value);
    this.x.grad += (s * (1 - s)) * this.out.grad
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
  Sigmoid,
  Constant,
  Variable,
  Gate,
  Sum,
  Times,
  Graph,
  Net
}

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
