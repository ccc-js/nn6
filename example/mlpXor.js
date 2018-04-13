const nn6 = require('../lib/nn6')

// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [0], [0], [1]], 10000) // and
// nn6.GradientLearning(new Net1(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]], 10000) // or
// 前兩個 and, or 成功，最後的 xor 加入動量後才成功！
nn6.GradientLearning(new nn6.net.Mlp2(), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 10000) // xor

