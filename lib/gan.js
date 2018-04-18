const Nd = require('./node')
const U = require('./util')
const N = require('./net')

/*
Steps to train a GAN
Step 1: Define the problem. Do you want to generate fake images or fake text. 
  Here you should completely define the problem and collect data for it.
Step 2: Define architecture of GAN. 
  Define how your GAN should look like. 
  Should both your generator and discriminator be multi layer perceptrons, 
  or convolutional neural networks? This step will depend on what problem you are trying to solve.
Step 3: Train Discriminator on real data for n epochs. 
  Get the data you want to generate fake on and train the discriminator to correctly predict them as real. 
  Here value n can be any natural number between 1 and infinity.
Step 4: Generate fake inputs for generator and train discriminator on fake data.
  Get generated data and let the discriminator correctly predict them as fake.
Step 5: Train generator with the output of discriminator.
  Now when the discriminator is trained, you can get its predictions and
  use it as an objective for training the generator. Train the generator to fool the discriminator.
Step 6: Repeat step 3 to step 5 for a few epochs.
Step 7: Check if the fake data manually if it seems legit. 
  If it seems appropriate, stop training, else go to step 3. 
  This is a bit of a manual task, as hand evaluating the data is the best way to check the fakeness. 
  When this step is over, you can evaluate whether the GAN is performing well enough.
*/

// 問題是， Generator 最後還是過度收斂在單一個點上，不管甚麼輸入，通通都說是那個點？??
// 這樣錯誤率是小了，但是整體卻沒有變好。
// 是否要加入 facts 反傳播給 Generator 去學習呢？這樣必須回饋 Discriminator 的梯度給 Generator 的 output ???
class Generator {
  constructor(gate) {
    this.gate = gate
  }
  generate(size) {
    // console.log('Generator.genrate:')
    this.vectors = []
    this.outputs = []
    let iLen = this.gate.inputs.length
    for (let i=0; i<size; i++) {
      let vector = this.gate.randomInputs() // U.repeats(iLen, ()=>U.rand(-1.0, 1.0))
      // console.log('  vector=%j', vector)
      Nd.setValues(this.gate.inputs, vector)
      this.gate.forward()
      // console.log('  out=%j', this.gate.out)
      // console.log('    list(a,b,x,y,s)=%s', this.gate.net.list(['a', 'b', 'x', 'y', 's']))
      this.vectors.push(vector)
      this.outputs.push(Nd.getValues(this.gate.out))
      // console.log('  outputs=%j', this.outputs[i])
    }
    return this.outputs
  }
  learn(samples, scores, loops=100) {
    // console.log('Generator.learn:')    
    let sLen = scores.length
    let average = U.mean(scores)
    for (let loop=0; loop<loops; loop++) {
      for (let i=0; i<sLen; i++) {
        this.gate.net.resetGrad()
        // console.log('  learn1:out=%j', this.gate.out)
        for (let o of this.gate.out) o.grad = scores[i]  - average // 用 0.5 的原因是 sigmoid 函數導致 scores[i] 的值在 0-1 之間
        // console.log('  learn2:out=%j', this.gate.out)
        // Nd.setValues(this.gate.out, scores[i])
        this.gate.backward()
        // console.log('learn:%j', this.gate.net.vars)
        this.gate.net.adjust() // -1*this.step, this.moment
      }  
    }
  }
}

class Discriminator {
  constructor(gate, facts) {
    this.gate = gate
    this.facts = facts
    console.log('facts = %j', facts)
    this.negatives = U.repeats(facts.length * 2, ()=>gate.randomInputs(-5, 5))
    console.log('this.negatives=%j', this.negatives)
    // process.exit(1)
  }
  evaluate(samples) {
    let scores = []
    for (let i=0; i<samples.length; i++) {
      Nd.setValues(this.gate.inputs, samples[i])
      this.gate.forward()
      // console.log('  ' + i + ' => ' + this.gate.net.dump())
      scores.push(this.gate.out[0].value)
    }
    return scores
  }
  learn(samples, score, tag, logging) {
    if (logging) console.log('  D.learn:', tag)
    let sLen = samples.length
    let out = this.gate.out
    for (let i=0; i<sLen; i++) {
      Nd.setValues(this.gate.inputs, samples[i])
      this.gate.forward()
      if (logging) console.log('    samples=%j out=%j score=%j', samples[i], out[0].value, score)
      this.gate.net.resetGrad()
      out[0].grad = score - out[0].value
      // Nd.setValues(this.gate.out, score)
      this.gate.backward()
      this.gate.net.adjust()
    }
  }
  discriminate(gens, facts, loops=100) {
    for (let i=0; i<loops; i++) {
      let logging = (i==99)
      this.learn(this.negatives, 0, 'negatives', logging) // Discriminator 學習評價負面樣本為 0
      this.learn(gens, 0.3, 'gens', logging)  // Discriminator 學習評價電腦樣本為 0.3
      this.learn(facts, 1, 'facts', logging) // Discriminator 學習評價人類樣本為 1
    }
  }
}

class Gan extends N.Net {
  constructor(facts) {
    super()
    this.facts = facts
    this.sampleSize = 10
  }
  learn(maxLoops) {
    let g = this.generator, d=this.discriminator
    // this.generator.init()
    // let gens = g.randomInputs
    let samples = g.generate(this.sampleSize) // Generator 產生初始樣本
    for (let loop = 0; loop < maxLoops; loop++) {
      // if (loop < 3) gens = gens.concat(samples)
      let scores = d.evaluate(samples) // Discriminator 評價新樣本
      console.log('%d:', loop)
      this.gunDump(g.vectors, samples, scores)
      g.learn(samples, scores, 10) // Generator 根據 Discriminator 的評價進行學習
      samples = g.generate(this.sampleSize) // Generator 產生新一輪的樣本
      // d.discriminate(samples, this.facts, 100)
      d.discriminate(samples, this.facts, 100)
      // break
    }
  }
  gunDump(vectors, samples, scores) {
    for (let i=0; i<vectors.length; i++) {
      console.log('  %d:v=%j x=%j s=%d', i, vectors[i], samples[i], scores[i])
    }
  }
}

const Mlp = require('./mlp').Mlp

class Gan2 extends Gan {
  constructor(facts) {
    super(facts)
    let {a,b,x,y,s} = this.addVariables(['a','b','x','y','s'])
    let mlpG = new Mlp([a, b], [3], [x, y], this)
    // let mlpG = new Mlp([a, b], [3,3], [x, y], this)
    this.generator = new Generator(mlpG)
    let mlpD = new Mlp([x, y], [3], [s], this)
    // let mlpD = new Mlp([x, y], [3,3], [s], this)
    this.discriminator = new Discriminator(mlpD, facts)
    this.setDumpVariables(['a', 'b', 'x', 'y', 's'])
  }
}

module.exports = {
  Discriminator,
  Generator,
  Gan,
  Gan2
}