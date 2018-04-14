const nn6 = require('../lib/nn6')

let rnn = new nn6.net.Rnn01()
nn6.GradientLearning(rnn, [[0], [0], [1], [0], [0], [1], [0], [0], [1]], [[0], [1], [0], [0], [1], [0], [0], [1], [0] ], 2000)
rnn.generate([0], 10)
/*
let o = 0
for (let i=0; i<10; i++) {
  let out = rnn.activate([o])
  o = (out[0].value >= 0.4) ? 1 : 0
  console.log('o=%d', o)
}
*/
