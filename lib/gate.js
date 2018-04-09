var D = require('./diff')

class IGate {
  constructor(inputs, out) {
    this.inputs = inputs
    this.out = out
  }
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
  Sigmoid,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Exp,
  Sum,
  Times
}
