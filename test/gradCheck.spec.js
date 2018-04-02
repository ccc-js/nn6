const nn6 = require('../lib/nn6')
var ok = require('assert').ok

let step = 0.000001

function checkGradient(gateClass, options = {}) {
  let G = nn6.Graph
  let {x, y, f} = G.newVariables(['x', 'y', 'f'])
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
      checkGradient(nn6.Sum)
    })
    it('check Times', function() {
      checkGradient(nn6.Times)
    })
    it('check Neg', function() {
      checkGradient(nn6.Neg)
    })
    it('check Rev', function() {
      checkGradient(nn6.Rev)
    })
    it('check Exp', function() {
      checkGradient(nn6.Exp)
    })
    it('check Relu', function() {
      checkGradient(nn6.Relu)
    })
    it('check Sigmoid', function() {
      checkGradient(nn6.Sigmoid)
    })
    it('check Add', function() {
      checkGradient(nn6.Add)
    })
    it('check Sub', function() {
      checkGradient(nn6.Sub, {checky: true})
    })
    it('check Mul', function() {
      checkGradient(nn6.Mul)
    })
    it('check Div', function() {
      checkGradient(nn6.Div, {checky: true})
    })
    it('check Pow', function() {
      checkGradient(nn6.Pow, {checky: true})
    })
  })
})
