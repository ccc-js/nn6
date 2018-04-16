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

function setInputs(nodes, values) {
  for (let i=0; i < values.length; i++) nodes.value = values[i]
}

function getOutputs(nodes) {
  let o = []
  for (let i=0; i < nodes.length; i++) o.push(nodes[i].value)
  return o
}

class Generator  {
  constructor(net) {
    this.net = net
  }
  generate(sampleSize) {
    for (let i=0; i<sampleSize; i++) {
      let vector = U.randomList(this.vLen, -1.0, 1.0)
      setInputs(this.inputs, vector)
      this.net.forward()
      return getOutputs(this.out)
    }
  }
  learn(samples, scores) {

  }
}

class Discriminator {
  constructor(net, facts) {
    this.net = net
    this.facts = facts
  }
  evaluate(samples) {
    let scores = []
    for (let i=0; i<samples.length; i++) {
      setInputs(this.inputs, samples[i])
      this.forward()
      scores.push(this.out.value)
    }
    return scores
  }
  learn(samples, facts) {

  }
}

let sampleSize = 10

class Gan {
  constructor() {}
  learn(facts) {
    this.generator.init()
    let samples = this.generator.generate() // Generator 產生初始樣本
    for (let loop = 0; loop < maxLoops; loop++) {
      let scores = this.discriminator.evaluate(samples) // Discriminator 評價新樣本
      this.generator.learn(samples, scores) // Generator 根據 Discriminator 的評價進行學習
      let samples = this.generator.generate(sampleSize) // Generator 產生新一輪的樣本
      this.discriminator.learn(samples, facts) // Discriminator 學習區分新樣本與標準答案
    }
  }
}

