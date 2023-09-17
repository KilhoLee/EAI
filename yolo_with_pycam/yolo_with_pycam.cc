/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "yolo_with_pycam.h"
#include <raspicam/raspicam_cv.h>
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>

using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  
  // (1) Pycam setting
  raspicam::RaspiCam_Cv camera;
  camera.set(cv::CAP_PROP_FORMAT, CV_8UC3);
  camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  if (!camera.open()) {
    cerr << "Error opening the camera" << endl;
    return 1;
  }
  while (true) {
    // (2) Load image from Pycam
    cv::Mat image;
    camera.grab();
    camera.retrieve(image);
    cv::imshow("Yolo example with Pycam", image);
    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }
    vector<cv::Mat> input;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(416, 416));
    input.push_back(image);
   
    // (3) Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // (4) Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // (5) Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");

    // (6) Push image to input tensor
    auto input_tensor = interpreter->typed_input_tensor<float>(0);
    for (int i=0; i<416; i++){
      for (int j=0; j<416; j++){   
        cv::Vec3b pixel = input[0].at<cv::Vec3b>(i, j);
        *(input_tensor + i * 416*3 + j * 3) = ((float)pixel[0])/255.0;
        *(input_tensor + i * 416*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
        *(input_tensor + i * 416*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
      }
    }
    // (7) Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");

    // (8) Output parsing
    TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
    TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
    yolo_output_parsing(cls_tensor, loc_tensor);

    // (9) Output visualize
    yolo_output_visualize(image);

    char key = cv::waitKey(1);
    if (key == 'q') {
        break;
    }
  }
  // (10) release
  camera.release();
	cv::destroyAllWindows();
  return 0;
}

  
