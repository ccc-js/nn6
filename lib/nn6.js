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

// const step = -0.01
// const step = 0.1

function GradientDescendent (net, maxLoops) {
  let opt = new Optimizer(net, ()=>net.gd())
  opt.run(maxLoops)
}
/*
function GradientLearning (net, inputs, facts, maxLoops) {
  for (let loop = 0; loop < maxLoops; loop++) {
    net.learnAll(loop, inputs, facts, step)
  }
}
*/

function GradientLearning (net, inputs, facts, maxLoops) {
  for (let loop = 0; loop < maxLoops; loop++) {
    let energy = 0
    if (loop % 100===0) console.log('%d:', loop)
    for (let i = 0; i < inputs.length; i++) {
      energy += net.learn(inputs[i], facts[i]) // step = -0.1
      net.forward()
      if (loop % 100===0) console.log('  ' + i + ' => ' + net.dump())
    }
    if (loop % 100===0) console.log('  ==> energy = %d', energy)
  }
}

module.exports = {
  node:N,
  gate:G,
  net:require('./net'),
  Optimizer,
  GradientDescendent,
  GradientLearning
}
