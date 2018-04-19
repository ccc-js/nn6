const Nd = require('./node')
const G = require('./gate')
const N = require('./net')

class Cnn extends N.NetGate {

}

// 參考： http://www.cnblogs.com/tornadomeet/p/3428843.html
// https://github.com/karpathy/convnetjs/blob/master/src/convnet_layers_nonlinearities.js
// 梯度計算： https://www.quora.com/How-are-gradients-computed-through-max-pooling-or-maxout-units-in-neural-networks
class MaxoutLayer extends G.Gate {
  forward () {
    let x = N.getValues(this.inputs)
    let o = F.max(x)
    N.setValues(this.out, o)
  }
  backward () {
    let iLen = this.inputs.length
    let o = F.max(x)
    for (let i=0; i<iLen; i++) {
      if (this.inputs[i].value === o) {
        this.inputs[i].grad = this.out.grad // 最大值的那個輸入之梯度，就是輸出的梯度
      }
    }
  }
}

class ConvGate extends VDot {
  constructor (input, out, net, filter) {
    super([input, filter], out, net)
  }
}

class ConvLayer extends NetGate {
  constructor (im, om, conv) {
    super()
    this.stride = 1
    this.im = im
    this.om = om
    this.conv = conv
  }
  forward () {
    let irows = im.length, icols=im[0].length, stride=this.stride
    let orows = om.length, ocols=om[0].length
    let cw = this.conv.width, ch = this.conv.height
    for (let or=0; or<irows; or++, ir += stride) {
      for (let oc=0; oc<cols; oc++, ic += stride) {
        let mlist = subm2list(im, ir, ic, cw, ch)
        N.setValues(conv.inputs, mlist)
        conv.forward()
        om[or][oc] = conv.out[0].value
      }
    }
  }
  backward () {
    let irows = im.length, icols=im[0].length, stride=this.stride
    let orows = om.length, ocols=om[0].length
    for (let or=0; or<irows; or++, ir += stride) {
      for (let oc=0; oc<cols; oc++, ic += stride) {
        conv.out[0].grad = dm[or][oc] 
        conv.backward()
      }
    }
  }
}

module.exports = {
  Cnn,
  ConvGate,
  ConvLayer,
  MaxoutLayer,
}