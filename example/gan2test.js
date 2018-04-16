const nn6 = require('../lib/nn6')

const gan2 = new nn6.gan.Gan2([[0, 0], [0.3, 0.3], [0.7, 0.7], [1, 1]])
gan2.learn(1000)
