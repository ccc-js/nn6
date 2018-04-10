
var constMap = {}

class Constant {
  constructor(value) {
    let c = constMap[value.toString()]
    if (c != null) return c
    this.val = value // 前向函數執行結果 Gate Output 為常數
  }
  adjust() {} // 常數不能修改
  get value() { return this.val }
  set value(c) { console.log('Constant cannot be set value') }
  get grad() { return 0 } // 常數的反向傳播梯度 Gradient = 0
  set grad(c) { /* console.log('Constant cannot be set grad') */ }
}

const moment = 0.01

class Variable {
  constructor(value, grad) {
    this.value = value || (Math.random() - 0.5) // 0.7 // 前向函數執行結果 Gate Output
    this.grad = grad || 0 // 反向傳播梯度 Gradient
    this.change = this.grad
  }
  adjust(step) {
    // 考慮動量 moment 與改變率 rate 的公式 (這個有錯，因為 learn 每次樣本都不一樣)
    this.value += step * this.grad + moment * this.change
    this.change = this.grad
    // 單純靠梯度的公式
    // this.value += step * this.grad
  }
}

function newConstants(list) {
  let a = []
  for (let name of list) {
    a.push(new Constant(name))
  }
  return a
}

function newVariables(list) {
  var d = {}
  for (let name of list) {
    d[name] = new Variable(0)
  }
  return d
}


module.exports = {
  Constant,
  Variable,
  newVariables,
  newConstants,
  PI: new Constant(Math.PI),
  E: new Constant(Math.E),
  C1: new Constant(1),
  C2: new Constant(2)
}
