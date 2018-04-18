function squareError(p, q) {
  let error = 0
  for (let i=0; i<p.length; i++) {
    let diff = p[i] - q[i]
    error += diff * diff
  }
  return error
}

function crossEntropy(p, q) {
  let loss = 0
  for (let i=0; i<p.length; i++) {
    loss += -1 * p[i] * Math.log (q[i])
  }
  return loss
}

module.exports = {
  squareError,
  crossEntropy,
}