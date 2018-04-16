function repeats(n, f) {
  let list = []
  for (let i=0; i<n; i++) list.push(f())
  return list
}

function rand(a, b) {
  return a + Math.random() * (b-a)
}

function randInt(a, b) {
  return Math.floor(rand(a, b))
}

module.exports = {
  repeats,
  rand,
  randInt,
}