const nn6 = require('../lib/nn6')
const G = nn6.gate
const ok = require('assert').ok
const step = 0.000001

function checkGradient(gateClass, options = {}) {
  let {x, y, f} = nn6.node.newVariables(['x', 'y', 'f'])
  x.value = 5.0
  y.value = 3.0
  f.grad = 1.0
  let gate = new gateClass([x, y], f)
  gate.forward()
  let fxy = f.value
  gate.backward()
  // check x diff
  x.value += step
  gate.forward()
  let fxpy = f.value
  let gx = (fxpy - fxy) / step
  console.log('x.grad = %d gx=%d', x.grad, gx)
  ok(Math.abs(gx - x.grad) < 0.01)
  // check y diff
  if (!options.checky) return
  x.value -= step
  y.value += step
  gate.forward()
  let fxyp = f.value
  let gy = (fxyp - fxy) / step
  console.log('y.grad = %d gy=%d', y.grad, gy)
  ok(Math.abs(gy - y.grad) < 0.01)
}

describe('nn6', function() {
  describe('gradient check', function() {
    it('check Sum', function() {
      checkGradient(G.Sum)
    })
    it('check Times', function() {
      checkGradient(G.Times)
    })
    it('check Neg', function() {
      checkGradient(G.Neg)
    })
    it('check Rev', function() {
      checkGradient(G.Rev)
    })
    it('check Exp', function() {
      checkGradient(G.Exp)
    })
    it('check Relu', function() {
      checkGradient(G.Relu)
    })
    it('check Sigmoid', function() {
      checkGradient(G.Sigmoid)
    })
    it('check Add', function() {
      checkGradient(G.Add)
    })
    it('check Sub', function() {
      checkGradient(G.Sub, {checky: true})
    })
    it('check Mul', function() {
      checkGradient(G.Mul)
    })
    it('check Div', function() {
      checkGradient(G.Div, {checky: true})
    })
    it('check Pow', function() {
      checkGradient(G.Pow, {checky: true})
    })
    it('check NPow', function() {
      checkGradient(G.NPow(2))
    })
  })
})
