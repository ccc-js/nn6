const G = require('./gate')
const N = require('./net')

class Rnn extends N.NetGate {
  activate(input) {
    for (let i=0; i<input.length; i++) this.inputs[i].value = input[i]
    this.forward()
    return this.out
  }
}

class XhConnect extends N.NetGate {
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
      new N.Connect([x], wx, net, wMin, wMax, null),
      new N.Connect([h], uh, net, wMin, wMax, null),
      new N.Layer([wx, uh, b], f, G.Sum),
      new N.Layer([f], out, Sig),
    ]
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
      new N.Connect([hidden], out, net, -1.0, 1.0)
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
      new N.Layer([i, ci], ic, G.Mul),
      new N.Layer([f, c], fc, G.Mul),
      new N.Layer([ic, fc], c, G.Add),
      new N.Layer([c], hi, G.Sigmoid),
      new N.Layer([out, hi], h, G.Mul),
    ]
  }
}

class RnnNet extends N.Net {
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
  Rnn,
  SimpleRnn,
  Lstm,
  RnnNet,
  Rnn01,
  Lstm01
}