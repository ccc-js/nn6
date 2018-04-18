var F = module.exports = {}

F.near = function (a, b, diff=0.000001) {
  return Math.abs(a - b) < diff
}

F.neg = function (x) {
  return -x
}

F.dneg = function (x) {
  return -1
}

F.rev = function (x) {
  return 1.0 / x
}

F.drev = function (x) {
  return -1 / (x*x)
}

F.exp = function (x) {
  return Math.exp(x)
}

F.dexp = function (x) {
  return Math.exp(x)
}

// 問題是，當所有的輸出值都小於 0，就會都被 Relu 截掉，於是變成 [0,0,0....]
// 所以通常要改用 leakyRelu, 請看後面的 leakyRelu 函數
F.relu = function (x) {
  return x > 0 ? x : 0.0
}

F.drelu = function (x) {
  return x > 0 ? 1 : 0
}

// 問題是，當所有的輸出值都小於 0，就會都被 Relu 截掉，於是變成 [0,0,0....]
// 所以 relu 通常要改用這個 leakyRelu, 即使小於 0 也會給一個很小的梯度。
const leaky = 0.01
F.leakyRelu = function (x) {
  return x > 0 ? x : leaky * x
}

F.dleakyRelu = function (x) {
  return x > 0 ? 1 : leaky
}

F.sigmoid = function (x) {
  return 1 / (1 + Math.exp(-x))
}

F.dsigmoid = function (x) {
  var s = F.sigmoid(x)
  return s * (1 - s)
}

F.tanh = function (x) {
  // return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
  return Math.tanh(x)
}

F.dtanh = function (x) {
  return 1.0 - x*x
}

F.pow = function (x, y) {
  return Math.pow(x, y)
}

F.dpow = function (x, y) {
  return y * Math.pow(x, y-1)
}

F.dpowy = function (x, y) {
  return Math.pow(x, y) * Math.log(x)
}

F.add = function (x, y) {
  return x + y
}

F.dadd = function (x, y) {
  return 1
}

F.sub = function (x, y) {
  return x - y
}

F.dsub = function (x, y) {
  return 1
}

F.dsuby = function (x, y) {
  return -1
}

F.mul = function (x, y) {
  return x * y
}

F.dmul = function (x, y) {
  return y
}

F.div = function (x, y) {
  return x / y
}

F.ddiv = function (x, y) {
  return 1 / y
}

F.ddivy = function (x, y) {
  return -x / (y * y)
}

F.max = function (list) {
  let r = Number.MIN_VALUE
  for (let x of list) {
    if (x > r) r = x
  }
  return r
}

// 這版的 softmax 會有 overflow, underflow 的危險 (只要總和超過幾百，或者小於幾百，就會爆了)，改一下！
// 參考： http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
F.softmax = function (list) {
  let sum = 0, r = [], e = []
  let max = F.max(list)
  for (let i=0; i<list.length; i++) {
    e[i] = Math.exp(list[i]-max)
    sum += e[i]
  }
  for (let i=0; i<list.length; i++) {
    r.push(e[i]/sum)
  }
  // console.log('Softmax: list=%j r=%j', list, r)
  return r
}


