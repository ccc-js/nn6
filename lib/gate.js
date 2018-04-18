var F = require('./func')
var U = require('./util')

class IGate {
  constructor(inputs, out) {
    this.inputs = inputs
    this.out = out
  }
  randomInputs (a=-1, b=1) { return U.repeats(this.inputs.length, ()=>U.rand(a, b)) }
  forward () { throw Error('IGate:forward not defined!') }
  backward () { throw Error('IGate:backward not defined!') }
  output(v) {
    this.out.value = v
    return this.out
  }
}

class Gate extends IGate {
  constructor(inputs, out, f, dfx, dfy) {
    super(inputs, out)
    this.f = f
    this.dfx = dfx
    this.dfy = dfy
  }
  forward () {
    let x = this.inputs[0], y = this.inputs[1], f = this.f
    let vy = (y) ? y.value : undefined 
    let o = f(x.value, vy)
    this.output(o)
  }
  backward () {
    let x = this.inputs[0], y = this.inputs[1], dfx = this.dfx, dfy = this.dfy, out = this.out
    let vy = (y) ? y.value : undefined 
    x.grad += dfx(x.value, vy) * out.grad
    if (this.dfy != null) {
      y.grad += dfy(x.value, vy) * out.grad
    } else if (y != null) {
      // 注意：只在對稱的情況下可以使用，例如 x+y, x*y, 在不對稱的情況不能使用，例如: x-y, x^y. 
      y.grad += dfx(vy, x.value) * out.grad
    }
  }
}

class CGate extends IGate {
  constructor(inputs, out) {
    super(inputs, out)
    this.gates = []
  }
  forward () {
    let gates = this.gates, len = gates.length
    for (let i = 0; i < len; i++) {
      gates[i].forward()
    }
  }
  backward () {
    let gates = this.gates, len = gates.length
    for (let i = len - 1; i >= 0; i--) {
      gates[i].backward()
    }
  }
}

// Uniary Operation
class Neg extends Gate {
  constructor(inputs, out) { super(inputs, out, F.neg, F.dneg) }
}

class Rev extends Gate {
  constructor(inputs, out) { super(inputs, out, F.rev, F.drev) }
}

class Relu extends Gate {
  constructor(inputs, out) { super(inputs, out, F.relu, F.drelu) }
}

class LeakyRelu extends Gate {
  constructor(inputs, out) { super(inputs, out, F.leakyRelu, F.dleakyRelu) }
}

class Sigmoid extends Gate {
  constructor(inputs, out) { super(inputs, out, F.sigmoid, F.dsigmoid) }
}

class Tanh extends Gate {
  constructor(inputs, out) { super(inputs, out, F.tanh, F.dtanh) }
}

class Exp extends Gate {
  constructor(inputs, out) {
    super(inputs, out, F.exp, F.dexp)
  }
}

function fn(f, n) {
  return function(x) { return f(x, n) }
}

class Pow extends Gate {
  constructor(inputs, out) {
    super(inputs, out, F.pow, F.dpow, F.dpowy)
  }
}

function NPow(n) {
  return function(inputs, out) {
    return new Pow(inputs, out, n)
  }
}

// Binary Operation
class Add extends Gate {
  constructor(inputs, out) { super(inputs, out, F.add, F.dadd) }
}

class Sub extends Gate {
  constructor(inputs, out) { super(inputs, out, F.sub, F.dsub, F.dsuby) }
}

class Mul extends Gate {
  constructor(inputs, out) { super(inputs, out, F.mul, F.dmul) }
}

class Div extends Gate {
  constructor(inputs, out) { super(inputs, out, F.div, F.ddiv, F.ddivy) }
}

// N-ary Operation
class Sum extends Gate {
  forward () {
    let s = 0
    for (let x of this.inputs) s += x.value
    this.output(s)
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
    this.output(s)
  }
  backward () {
    let s = 1
    for (let x of this.inputs) s *= x.value
    for (let x of this.inputs) {
      x.grad += (s / x.value) * this.out.grad
    }
  }
}

module.exports = {
  IGate,
  Gate,
  CGate,
  Neg,
  Rev,
  Relu,
  LeakyRelu,
  Sigmoid,
  Tanh,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  NPow,
  Exp,
  Sum,
  Times
}
