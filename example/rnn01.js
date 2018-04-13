const nn6 = require('../lib/nn6')

nn6.GradientLearning(new nn6.net.Rnn01(), [[0], [0], [1], [0], [0], [1], [0], [0], [1]], [[0], [1], [0], [0], [1], [0], [0], [1], [0] ], 2000)

