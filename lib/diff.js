var D = module.exports = {}

D.neg = function (x) {
  return -x
}

D.dneg = function (x) {
  return -1
}

D.rev = function (x) {
  return 1.0 / x
}

D.drev = function (x) {
  return -1 / (x*x)
}

D.exp = function (x) {
  return Math.exp(x)
}

D.dexp = function (x) {
  return Math.exp(x)
}

D.relu = function (x) {
  return x > 0 ? x : 0.0
}

D.drelu = function (x) {
  return x > 0 ? 1 : 0
}

D.sigmoid = function (x) {
  return 1 / (1 + Math.exp(-x))
}

D.dsigmoid = function (x) {
  var s = D.sigmoid(x)
  return s * (1 - s)
}

D.tanh = function (x) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
  // return Math.tanh(x)
}

D.dtanh = function (x) {
  return 1.0 - x*x
}

D.pow = function (x, y) {
  return Math.pow(x, y)
}

D.dpow = function (x, y) {
  return y * Math.pow(x, y-1)
}

D.dpowy = function (x, y) {
  return Math.pow(x, y) * Math.log(x)
}

D.add = function (x, y) {
  return x + y
}

D.dadd = function (x, y) {
  return 1
}

D.sub = function (x, y) {
  return x - y
}

D.dsub = function (x, y) {
  return 1
}

D.dsuby = function (x, y) {
  return -1
}

D.mul = function (x, y) {
  return x * y
}

D.dmul = function (x, y) {
  return y
}

D.div = function (x, y) {
  return x / y
}

D.ddiv = function (x, y) {
  return 1 / y
}

D.ddivy = function (x, y) {
  return -x / (y * y)
}
