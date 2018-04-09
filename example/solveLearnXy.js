const nn6 = require('../lib/nn6')

// 2x+y=3
//  x+y=2
// 解答： x=1, y=1
class Net1 extends nn6.net.Net {
  constructor () {
    super()
    let {a, b, c, d, x, y, r1, r2, _2x, f} = this.addVariables(['a', 'b', 'c', 'd', 'x', 'y', 'r1', 'r2', '_2x', 'f'])
    this.setDumpVariables(['x', 'y', 'r1', 'r2', 'f'])
    let [c1, c2, c3] = nn6.node.newConstants([1, 2, 3])
    this.out = f
    this.gates = [
      new nn6.net.MDot([[[c2, c1], [c1, c1]], [x, y]], [r1, r2], this),
    ]
  }
}

nn6.GradientLearning(new Net1(), [[2, 1], [1, 1]], [3, 2],  1000)

