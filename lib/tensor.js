var V = require('./vector')

var T = module.exports = {}

var fop = {
  '+' : (a,b) => a+b,
  '-' : (a,b) => a-b,
  '*' : (a,b) => a*b,
  '/' : (a,b) => a/b,
}

var vop = {
  '+' : V.add,
  '-' : V.sub,
  '*' : V.mul,
  '/' : V.div,
}

T.op = function (op, a, b) {
  if (typeof a === 'number') {
    return fop[op](a, b)
  } else {
    return vop[op](a, b)
  }
}
