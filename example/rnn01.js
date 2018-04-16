const nn6 = require('../lib/nn6')

let rnn = new nn6.rnn.Rnn01()
nn6.gradientLearning(rnn, [[0], [0], [1], [0], [0], [1], [0], [0], [1]], [[0], [1], [0], [0], [1], [0], [0], [1], [0] ], 2000)
rnn.generate([0], 10)

