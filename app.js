const express = require('express');

const tf = require('@tensorflow/tfjs-node');

const helmet = require('helmet');

const compression = require('compression');

const formidable = require('formidable');
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');
const fs = require('fs');
const sharp = require('sharp');

const app = express();

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

// app.get('/', function(req, res, next) {
//     res.render('index', { title: 'Express' });
//   });

let mirNetModel;
let modelInfo;

async function loadModel() {
  try {
    // Warm up the model
    if (!mirNetModel) {

      modelInfo = await tf.node.getMetaGraphsFromSavedModel('./model');

      // Load the TensorFlow SavedModel through tfjs-node API. You can find more
      // details in the API documentation:
      // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
      mirNetModel = await tf.node.loadSavedModel(
        './model/'
      );
      return mirNetModel;
    }
  } catch (error) {
    console.log(error);
  }
};

loadModel();
const predict = async (imgPath, responseImagePath) => {
  try {
    console.log("Loading Model");
    await loadModel();
    console.log("Inside predict");

    // image = new Uint8Array(image);
    // Decode the image into a tensor.
    let img = sharp(imgPath).removeAlpha().resize({ width: 512, height: 512, fit: "contain", }).png().toBuffer();
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

    fs.writeFileSync(responseImagePath, outputTensor);
    return(outputTensor, true);

} catch (error) {
    console.log(error);
    throw error;
}
};
const formOptions = {
  // uploadDir: path.join(__dirname, "uploads"),
  encoding: 'utf-8',
  keepExtensions: true,
  maxFileSize: 5 * 1024 * 1024,
  multiples: false,
};
// const form = new formidable.IncomingForm(formOptions);


app.post('/submit', (req, res) => {
  let form = new formidable.IncomingForm(formOptions);
  form.parse(req, async (err, fields, files) => {

    if (err) {
      res.send("Incorrect File Format");
      console.log('\n' + err + '\n');
    } else {
      // console.log("sending File: " + files.image.name);
      // res.sendFile(files.image.path);
      try {
        let toSend = await predict(files.image.path, path.join(__dirname, "public", "responseImages") + files.image.name);
        if(toSend === true) {
          res.sendFile(path.join(__dirname, "public", "responseImages") + files.image.name);
        } else {
          res.status(501).send(toSend);
        }
      } catch (err) {
        res.status(500).send(err);
      }
    }
  });
  const allowedFiles = ["image/jpg", "image/jpeg", "image/png"];
  try {
    form.on('fileBegin', function (name, file) {
      if (!allowedFiles.includes(file.type)) {
        // throw new Error("Incorrect File Type");
        form._error(new Error("Incorrect File Type"));
        return new Error("Incorrect File Type");
      } else {
        file.path = path.join(__dirname, "uploads") + "/" + Date.now() + "-" + file.name;
      }
    });
  } catch (err) {
    form._error(err);
    console.log(err);
    res.send("Incorrect File Type");
  }
});

module.exports = app;
