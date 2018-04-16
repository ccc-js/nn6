function rand(a, b) {
  return a + Math.random() * (b-a)
}

function randInt(a, b) {
  return Math.floor(rand(a, b))
}

module.exports = {
  rand,
  randInt,
}