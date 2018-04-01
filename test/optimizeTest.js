const nn6 = require('../lib/nn6')

// f(x,y) = x^2 + 2xy + y^2 + 2
// 在 x=0 y=0 時，有最小值 2
class Net1 extends nn6.Net {
  constructor () {
    super()
    let v = this.vars = nn6.Graph.newVariables(['x', 'y', 'xx', '_2xy', 'yy', 'f'])
    let [c2] = nn6.Graph.newConstants([2])
    this.inputs = [v.x, v.y]
    v.x.value = 1
    v.y.value = 2
    this.out = v.f
    this.gates = [
      new nn6.Times([v.x, v.x], v.xx),
      new nn6.Times([c2, v.x, v.y], v._2xy),
      new nn6.Times([v.y, v.y], v.yy),
      new nn6.Sum([v.xx, v._2xy, v.yy, c2], v.f)
    ]
    console.log('vars=%j', this.vars)
  }
}

let net = new Net1()
net.run(300)

  /*
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
    console.log('out=%d', this.out.value)
    this.resetGradient()
    this.out.grad = 1.0
    console.log('vars=%j', this.vars)
    this.backward()
    this.adjust(step)
  }
  run(maxLoops) {
    var lastValue = 9999
    for (let i = 0; i < maxLoops; i++) {
      this.gd(-0.01)
      // console.log('  inputs:%j', this.inputs)
      // console.log('out=%d', i, out.value)
      if (this.out.value > lastValue) break
      lastValue = this.out.value
    }
  }
  */
/*

var lastValue = 9999

for (let i=0; i<300; i++) {
  var out = net.gd(-0.01)
  // console.log('  x=%j y=%j', net.x, net.y)
  console.log('%d:x=%d y=%d out=%d', i, net.x.value, net.y.value, out.value)
  if (out.value > lastValue) break
  lastValue = out.value
}
function gradientDescendent(step) {
  let out = net.forward()
  // console.log('x=%j\ny=%j\nout=%j', net.x, net.y, out)
  net.x.grad = 0
  net.y.grad = 0
  out.grad = 1.0
  net.backward()
  net.adjust(step)
  return out
}
*/