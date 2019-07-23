import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://github.com/kelliesyhh/kerastotfjs/blob/master/model/model.json');

// list containing input for every layer
var input = []
// First Layer;s Input is Facial Image
input.push(tf.tidy(() => { return tf.expandDims(facialImage,0).asType('float32').div(255.0)}));
       
// passing Input to every layer and saving output as input for next layer    
for (var i = 1; i <= 12; i++) {
   input.push(genderAI.layers[i].apply(input[i-1]));
}
       
// Saving activationmaps (note that we are displaying activation map after applying max pool)     
const firstconvactivationmap = input[2];
const secondconvactivationmap = input[4];
const thirdconvactivationmap = input[6];