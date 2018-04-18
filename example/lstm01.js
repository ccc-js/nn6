const nn6 = require('../lib/nn6')

let lstm = new nn6.rnn.Lstm01()
nn6.gradientLearning(lstm, [[0], [0], [1], [0], [0], [1], [0], [0], [1]], [[0], [1], [0], [0], [1], [0], [0], [1], [0] ], 3000)
lstm.generate([0], 20)
