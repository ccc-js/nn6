const ok = require('assert').ok
var nj = require('numjs')

describe('tensor', function() {
  describe('numjs', function() {
    it('', function() {
      let a = nj.arange(15).reshape(3, 5)
      ok(a.get(1,1) === 6)
    })
  })
})
