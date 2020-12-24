const tf = require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
let mirNetModel;
let modelInfo;
const imageSize = 512;

async function loadModel() {
    try {
        // Warm up the model
        if (!mirNetModel) {

            modelInfo = await tf.node.getMetaGraphsFromSavedModel('./model');
            //   console.log("\n\n\n");
            //   console.log(modelInfo);
            //   console.log("\n\n\n");

            // Load the TensorFlow SavedModel through tfjs-node API. You can find more
            // details in the API documentation:
            // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
            mirNetModel = await tf.node.loadSavedModel(
                './model'
            );
            return mirNetModel;
        }
    } catch (error) {
        console.log(error);
        throw error
    }
}
const predict = async () => {
    try {
        console.log("Loading Model");
        await loadModel();
        console.log("Inside predict");

        // image = new Uint8Array(image);
        // Decode the image into a tensor.
        let img = sharp("Capture.PNG").removeAlpha().resize({ width: 512, height: 512, fit: "contain", }).png().toBuffer();
        img = new Uint8Array.from(img);
        let imageTensor = await tf.node.decodePng(img, 3);
        // imageTensor = tf.image.resizeBilinear(imageTensor, size = [imageSize, imageSize])
        console.log("after img 2 tensor");

        let input = imageTensor.expandDims(0);
        // [:,:,:3]
        // input = input[[,,], [,,], [,3]]
        console.log(input);

        // Feed the image tensor into the model for inference.
        const startTime = tf.util.now();
        input = tf.cast(input, "float32");

        let outputTensor = await mirNetModel.predict({ 'input_1': input });

        const endTime = tf.util.now();
        console.log(endTime - startTime);
        console.log("After Predict");

        console.log(outputTensor.add_171);

        outputTensor = outputTensor.add_171;
        outputTensor = tf.reshape(outputTensor, [512, 512, 3]);

        // outputTensor = new Uint8Array(outputTensor);
        outputTensor = await tf.node.encodePng(outputTensor);

        fs.writeFileSync("./NEW-Capture.png", outputTensor);

    } catch (error) {
        console.log(error);
    }
};

(async () => {
    console.log("Starting Execution");
    await predict();
    console.log("DONE");
})();