const nn6 = require('../lib/nn6')
const R = nn6.rgan
const Mlp = nn6.mlp.Mlp

const facts = [
  //  A B C D E F G     b3b2b1b0
  {i:[1,1,1,1,1,1,0], o:[0,0,0,0], t:'0'}, // 0
  {i:[0,1,1,0,0,0,0], o:[0,0,0,1], t:'1'}, // 1
  {i:[1,1,0,1,1,0,1], o:[0,0,1,0], t:'2'}, // 2
  {i:[1,1,1,1,0,0,1], o:[0,0,1,1], t:'3'}, // 3
  {i:[0,1,1,0,0,1,1], o:[0,1,0,0], t:'4'}, // 4
  {i:[1,0,1,1,0,1,1], o:[0,1,0,1], t:'5'}, // 5
  {i:[1,0,1,1,1,1,1], o:[0,1,1,0], t:'6'}, // 6
  {i:[1,1,1,0,0,0,0], o:[0,1,1,1], t:'7'}, // 7
  {i:[1,1,1,1,1,1,1], o:[1,0,0,0], t:'8'}, // 8
  {i:[1,1,1,1,0,1,1], o:[1,0,0,1], t:'9'}, // 9
]

class RGan7Seg extends R.RGan {
  constructor(facts) {
    super(facts)
    let {i0,i1,i2,a,b,c,d,e,f,g,b3,b2,b1,b0,s} = this.addVariables(['i0','i1','i2','a','b','c','d','e','f','g','b3','b2','b1','b0','s'])
    let mlpG = new Mlp([b3,b2,b1,b0,i0,i1,i2], [5], [a,b,c,d,e,f,g], this) // [5, 5]
    this.generator = new R.Generator(mlpG)
    let mlpD = new Mlp([b3,b2,b1,b0,a,b,c,d,e,f,g], [5], [s], this) // [5, 4]
    this.discriminator = new R.Discriminator(mlpD, facts)
    this.setDumpVariables(['s','b3','b2','b1','b0','a','b','c','d','e','f','g']) // ,
  }
}

const gan = new RGan7Seg(facts)
gan.learn(10000)