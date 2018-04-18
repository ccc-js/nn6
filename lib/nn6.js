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

function gradientLearning (net, inputs, facts, maxLoops, dumpPeriod = 100) {
  for (let loop = 0; loop < maxLoops; loop++) {
    if (loop % dumpPeriod===0) console.log('%d:', loop)
    let loss = 0
    for (let i = 0; i < inputs.length; i++) {
      // if (loop % dumpPeriod===0) console.log('  before:' + i + ' => ' + net.dump())
      // console.log('net.vars=%j', net.vars)
      loss += net.learn(inputs[i], facts[i]) // step = -0.1
      // console.log('   ===> energy[%d]=%d', i, energy)
      net.forward()
      if (loop % dumpPeriod===0) console.log(i + ' => ' + net.dump())
    }
    if (loop % dumpPeriod===0) console.log('  ==> loss = %d', loss)
    // process.exit(1)
  }
}

module.exports = {
  node: N,
  gate: G,
  Optimizer,
  gradientDescendent,
  gradientLearning,
  func: require('./func'),
  loss: require('./loss'),
  net: require('./net'),
  mlp: require('./mlp'),
  rnn: require('./rnn'),
  // gan: require('./gan'),
  rgan: require('./rgan'),
}
