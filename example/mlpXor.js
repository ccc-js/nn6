const nn6 = require('../lib/nn6')

// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [0], [0], [1]], 10000) // and
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]], 10000) // or
// 前兩個 and, or 成功，但最後的 xor 還是失敗！
nn6.GradientLearning(new nn6.net.Mlp2(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 1000) // xor

