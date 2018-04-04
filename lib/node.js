
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
  set grad(c) { }
}

class Variable {
  constructor(value, grad) {
    this.value = value || 0// 前向函數執行結果 Gate Output
    this.grad = grad || 0 // 反向傳播梯度 Gradient
  }
  adjust(step) {
    this.value += step * this.grad
  }
}

module.exports = {
  Constant,
  Variable,
  /*
  newVariables,
  newConstants,
  newVector,
  */
  PI: new Constant(Math.PI),
  E: new Constant(Math.E),
  C1: new Constant(1),
  C2: new Constant(2)
}
