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
  squareError, // 最小平方法
  regression: squareError, // Regression 就是最小平方法
  crossEntropy, // 通常搭配 Softmax, Relu 使用
}