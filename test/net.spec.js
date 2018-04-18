const nn6 = require('../lib/nn6')
const ok = require('assert').ok
const step = 0.000001

function checkGradient(netClass, options = {}) {
  let net = new netClass()
  let {x1, x2, f} = net.vars // net.addVariables(['x', 'y', 'f'])
  let x = x1, y=x2
  x.value = 0.8
  y.value = 0.6
  f.grad = 1.0
  net.forward()
  // console.log('net.vars=%j', net.vars)
  let fxy = f.value
  net.backward()
  // check x diff
  x.value += step
  net.forward()
  let fxpy = f.value
  let gx = (fxpy - fxy) / step
  console.log('x.grad = %d gx=%d', x.grad, gx)
  // check y diff
  if (!options.checky) return
  x.value -= step
  y.value += step
  net.forward()
  let fxyp = f.value
  let gy = (fxyp - fxy) / step
  console.log('y.grad = %d gy=%d', y.grad, gy)
  ok(Math.abs(gx - x.grad) < 0.01 && Math.abs(gy - y.grad) < 0.01)
}

describe('nn6', function() {
  describe('gradient check', function() {
    it('check Perceptron2', function() {
      checkGradient(nn6.mlp.Perceptron2, {checky: true})
    })
    it('check Mlp2', function() {
      checkGradient(nn6.mlp.Mlp2, {checky: true})
    })
  })
})
