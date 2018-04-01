class Constant {
  constructor(value) {
    this.val = value // 前向函數執行結果 Gate Output 為常數
    this.grad = 0   // 反向傳播梯度 Gradient = 0
  }
  get value() {
    return this.val
  }
  set value(v) {
    console.log('Constant cannot be set value')
  }
  get grad() {
    return 0
  }
  set grad(v) {
    // console.log('Constant cannot be set grad')
  }
}

class Variable {
  constructor(value, grad) {
    this.value = value // 前向函數執行結果 Gate Output
    this.grad = grad || 0.0 // 反向傳播梯度 Gradient
  }
}

/*
function newVariables(list) {
  let d = {}
  for (let v of list) {
    d[v] = new Variable(0)
  }
  return d
}

function newConstants(list) {
  let a = []
  for (let v of list) {
    a.push(new Constant(v))
  }
  return a
}
*/

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

/*
class Sigmoid extends Gate {
  sig(x) {
    return 1 / (1 + Math.exp(-x))
  }
  forward (x) {
    this.x = x
    this.out = new Variable(this.sig(this.x.value))
    return this.out
  }
  backward () {
    var s = this.sig(this.x.value);
    this.x.grad += (s * (1 - s)) * this.out.grad
  }
}
*/

class Graph {
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
}

class Net {
  constructor() { }
  forward () {
    for (let i = 0; i < this.gates.length; i++) {
      this.gates[i].forward()
    }
  }
  backward () {
    for (let i = this.gates.length - 1; i >=0; i--) {
      this.gates[i].backward()
    }
  }
  adjust (step) {
    for (let node of this.inputs) {
      node.value += step * node.grad
    }
  }
  resetGradient() {
    let nodes = Object.values(this.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  gd(step) {
    this.forward()
    this.resetGradient()
    this.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  run(maxLoops) {
    var lastValue = 9999
    for (let i = 0; i < maxLoops; i++) {
      this.gd(-0.01)
      // console.log('  inputs:%j', this.inputs)
      console.log('out=%d', i, this.out.value)
      if (this.out.value > lastValue) break
      lastValue = this.out.value
    }
  }
}

module.exports = {
  // Sigmoid,
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
