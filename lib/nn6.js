var N = require('./node')
var G = require('./gate')

class Optimizer {
  constructor(net, improver) {
    this.net = net
    this.improver = improver
  }
  run(maxLoops) {
    let lastValue = 9999
    let out = this.net.out
    for (let i = 0; i < maxLoops; i++) {
      this.improver()
      this.net.forward()
      console.log(i + ':' + this.net.dump())
      if (out.value > lastValue) break
      lastValue = out.value
    }
  }
}

function GradientDescendent (net, maxLoops) {
  let opt = new Optimizer(net, (step)=>net.gd(-0.01))
  opt.run(maxLoops)
}

module.exports = {
  node:N,
  gate:G,
  net:require('./net'),
  /*
  Layer,
  Net,
  */
  Optimizer,
  GradientDescendent
}
