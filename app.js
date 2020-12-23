const express = require('express');

const tf = require('@tensorflow/tfjs-node');

const helmet = require('helmet');

const compression = require('compression');

const formidable = require('formidable');
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');

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
  // Warm up the model
  if (!mirNetModel) {

    modelInfo = await tf.node.getMetaGraphsFromSavedModel('./model');

    // Load the TensorFlow SavedModel through tfjs-node API. You can find more
    // details in the API documentation:
    // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
    mirNetModel = await tf.node.loadSavedModel(
      './model'
    );
  }
}

const formOptions = {
  uploadDir: path.join(__dirname, "uploads"),
  encoding: 'utf-8',
  keepExtensions: true,
  maxFileSize: 5 * 1024 * 1024,
  multiples: false,
};
const form = new formidable.IncomingForm(formOptions);


app.post('/submit', (req, res) => {
  form.parse(req);
  const allowedFiles = ["image/jpg", "image/jpeg", "image/png"];
  form.on('fileBegin', function (name, file) {
    if (! allowedFiles.includes(file.type) ) {
      throw new Error("Incorrect File Type");
    }
    file.path = form.uploadDir + file.name + "-" + Date.now();
  });
  form.on('file', function (name, file) {
    console.log('Uploaded ' + file.name + "\tTo: " + file.path.split('\\').reduce((filename, str) => {
      if (str.includes(/jpg|jpeg|png/)) {
        return str
      }
    }));
    //  let returnData = sendinfo(file.path);
    //   console.log("RETURN DATA: \n"+returnData);
    //  res.send(returnData);
    // res.send(sendinfo(file.path));
    res.send(file.path + "<br>" + file.name, res);
  });
  form.on('error', err => {
    res.send("Incorrect File Format");
    console.log('\n' + err + '\n');
  });
});

module.exports = app;
