var node = require('./node')
var gate = require('./gate')
var layer = require('./layer')
// var V = require('./vector')
// var T = require('./tensor')

class Graph {
  constructor() {
    this.inputs = []
    this.gates = []
    this.vars = {}
    this.out = null
  }
  addVariables(list) {
    Object.assign(this.vars, node.newVariables(list))
    return this.vars
  }
  addInputs(inputs) {
    this.inputs = this.inputs.concat(inputs)
  }
  addGates(gates) {
    this.gates = this.gates.concat(gates)
  }
  setOutput(out) {
    this.out = out
  }
}

class Net {
  constructor(graph) {
    this.graph = graph
  }
  forward () {
    let gates = this.graph.gates
    for (let i = 0; i < gates.length; i++) {
      gates[i].forward()
    }
  }
  backward () {
    let gates = this.graph.gates
    for (let i = gates.length - 1; i >=0; i--) {
      gates[i].backward()
    }
  }
  adjust (step) {
    let inputs = this.graph.inputs
    for (let node of inputs) {
      node.value += step * node.grad
    }
  }
  resetGradient() {
    let nodes = Object.values(this.graph.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  gd(step) {
    this.forward()
    this.resetGradient()
    this.graph.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  run(maxLoops) {
    let lastValue = 9999
    let out = this.graph.out
    for (let i = 0; i < maxLoops; i++) {
      this.gd(-0.01)
      // console.log('  inputs:%j', this.inputs)
      console.log('%d: out=%d', i, out.value)
      if (out.value > lastValue) break
      lastValue = out.value
    }
  }
}
/*
class Optimizer {
  adjust (step) {
    let inputs = this.graph.inputs
    for (let node of inputs) {
      node.value += step * node.grad
    }
  }
  resetGradient() {
    let nodes = Object.values(this.graph.vars)
    for (let node of nodes) {
      node.grad = 0
    }
  }
  gd(step) {
    this.forward()
    this.resetGradient()
    this.graph.out.grad = 1.0
    this.backward()
    this.adjust(step)
  }
  run(maxLoops) {
    let lastValue = 9999
    let out = this.graph.out
    for (let i = 0; i < maxLoops; i++) {
      this.gd(-0.01)
      // console.log('  inputs:%j', this.inputs)
      console.log('%d: out=%d', i, out.value)
      if (out.value > lastValue) break
      lastValue = out.value
    }
  }
}
*/
module.exports = {
  node,
  gate,
  // Net
  Graph,
  Net
}
