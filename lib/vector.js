var V = module.exports = {}

V.new = function (n, value = 0) { return new Array(n).fill(0) }

V.resize = function (a, size) {
  var v = a.slice()
  for (var i = a.length; i < size; i++) {
    v.push(0)
  }
  return v
}

V.dot = function (x, y, isComplex = false) {
  let sum = 0
  let len = x.length
  for (var i = 0; i < len; i++) {
    if (!isComplex) {
      sum += x[i] * y[i] // 速度較快
    } else {
      sum = cadd(sum, cmul(x[i], y[i])) // 速度稍慢(不使用多型，否則會很慢)
    }
  }
  return sum
}

V.add = function (a, b) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] + b[i]
  }
  return r
}

V.sub = function (a, b) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] - b[i]
  }
  return r
}

V.mul = function (a, b) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] * b[i]
  }
  return r
}

V.div = function (a, b) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] / b[i]
  }
  return r
}