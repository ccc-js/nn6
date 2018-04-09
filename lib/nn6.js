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

const step = -0.01

function GradientDescendent (net, maxLoops) {
  let opt = new Optimizer(net, ()=>net.gd(step))
  opt.run(maxLoops)
}

function GradientLearning (net, inputs, facts, maxLoops) {
  for (let loop = 0; loop < maxLoops; loop++) {
    for (let i = 0; i < inputs.length; i++) {
      net.learn(inputs[i], facts[i], step)
      net.forward()
      console.log(i + ':' + net.dump())
    }
  }
}

module.exports = {
  node:N,
  gate:G,
  net:require('./net'),
  Optimizer,
  GradientDescendent
}
