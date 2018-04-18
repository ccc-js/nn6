const nn6 = require('../lib/nn6')
const F = nn6.func
const ok = require('assert').ok

describe('func', function() {
  describe('Softmax', function() {
    it('softmax([1000,1000])', function() {
      ok(F.near(F.softmax([1000, 1000])[0], 0.5))
    })
    it('softmax([1, 10,1000])', function() {
      ok(F.near(F.softmax([1, 10, 1000])[0], 0.0))
    })
    it('softmax([1, 10,1000])', function() {
      ok(F.near(F.softmax([1, 10, 1000])[0], 0.0))
    })
  })
})
