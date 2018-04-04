var M = module.exports = {}

M.transpose = function (m) {
  var r = []
  var rows = m.length
  var cols = m[0].length
  for (var j = 0; j < cols; j++) {
    var rj = r[j] = []
    for (var i = 0; i < rows; i++) {
      rj[i] = m[i][j]
    }
  }
  return r
}
