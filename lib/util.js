function repeats(n, f) {
  let list = []
  for (let i=0; i<n; i++) list.push(f())
  return list
}

function sum(list) {
  let s = 0
  for (let x of list) s+=x
  return s
}

function mean(list) {
  return sum(list)/list.length
}

function rand(a, b) {
  return a + Math.random() * (b-a)
}

function randInt(a, b) {
  return Math.floor(rand(a, b))
}

function randChoose(list) {
  return list[randInt(0, list.length)]
}

module.exports = {
  repeats,
  rand,
  randInt,
  randChoose,
  sum,
  mean,
}