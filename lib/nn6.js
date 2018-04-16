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

function gradientDescendent (net, maxLoops) {
  let opt = new Optimizer(net, ()=>net.gd())
  opt.run(maxLoops)
}

function gradientLearning (net, inputs, facts, maxLoops) {
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
  node: N,
  gate: G,
  Optimizer,
  gradientDescendent,
  gradientLearning,
  net: require('./net'),
  mlp: require('./mlp'),
  rnn: require('./rnn'),
  gan: require('./gan'),
}
