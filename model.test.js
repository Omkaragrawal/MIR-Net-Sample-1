const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
let mirNetModel;
let modelInfo;

async function loadModel() {
    try {
        // Warm up the model
        if (!mirNetModel) {

              modelInfo = await tf.node.getMetaGraphsFromSavedModel('./model');
              console.log("\n\n\n");
              console.log(await modelInfo);
              console.log("\n\n\n");

            // Load the TensorFlow SavedModel through tfjs-node API. You can find more
            // details in the API documentation:
            // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
            mirNetModel = await tf.node.loadSavedModel(
                './model'
            );
            return await mirNetModel;
        }
    } catch (error) {
        console.log(error);
    }
}
const predict = async () => {
    try {
        await loadModel();
        console.log("Inside predict");
        let image = fs.readFileSync("./uploads/1.png");
        image = new Uint8Array(image);
        // Decode the image into a tensor.
        const imageTensor = await tf.node.decodeImage(image);
        console.log("after img 2 tensor");
        const input = imageTensor.expandDims(0);

        // Feed the image tensor into the model for inference.
        const startTime = tf.util.now();
        let outputTensor = await mirNetModel.predict({ 'x': input });
        console.log("After Predict");
        fs.writeFile("./uploads/NEW-1.png", await outputTensor, (err) => {
            if (err) {
                console.log("ERR");
                console.log(err);
            } else {
                // Parse the model output to get meaningful result(get detection class and
                // object location).
                const endTime = tf.util.now();
                console.log(endTime - startTime);
            }
        });

    } catch (error) {
        console.log(error);
    }
};

(async () => {
    await predict();
    console.log("DONE");
})()



// fs.writeFileSync("./uploads/NEW-1.png", a);
// fs.readFileSync()