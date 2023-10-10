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
#include <memory>
#include <chrono>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "headers/edgetpu_c.h"
#include "opencv2/opencv.hpp"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.

using namespace std;
using namespace cv;


// MAKE SURE TO USE PROPER DIRECTORIES
#define IMAGENET_BANANA "/home/pi/EAI/imagenet_dataset/banana.jpg"
#define IMAGENET_ORANGE "/home/pi/EAI/imagenet_dataset/orange.jpg"
#define IMAGENET_LABELS "/home/pi/EAI/imagenet_dataset/imagenet_label.txt"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// Read image with opencv
void readImageCV(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}

	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
  cv::resize(cvimg, cvimg, cv::Size(224, 224)); //resize to 224 224
  
  cvimg.convertTo(cvimg, CV_8U);
  input.push_back(cvimg);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "minimal <model> <use tpu 0/1> <inference num>\n");
    return 1;
  }

  bool use_quantization = false;
  const char* filename = argv[1];
  bool use_tpu = std::stoi(argv[2]);

  int inference_num = std::stoi(argv[3]);

  if(use_tpu){
    std::cout << "Use TPU acceleration" << "\n";
  }
  else{
    std::cout << "No TPU acceleration" << "\n";
  }
  std::cout << "Inference " << inference_num << " times and get average latency" << "\n";
  

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Setup for Edge TPU device.
  if(use_tpu){
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    assert(num_devices > 0);
    const auto& device = devices.get()[0];

    // Create TPU delegate.
    auto* delegate =
      edgetpu_create_delegate(device.type, device.path, nullptr, 0);

    // Delegate graph.
      interpreter->ModifyGraphWithDelegate(delegate);
  }

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Variables to measure invoke latency.
  struct timespec begin, end;
  double latency = 0;

  
  // Read input image   
  vector<cv::Mat> inputs;
  readImageCV(IMAGENET_BANANA, inputs);
  readImageCV(IMAGENET_ORANGE, inputs);
  
  for(int seq=0; seq<inference_num; ++seq){
    // Fill input buffers
    // TODO(user): Insert code to fill input tensors.
    // Note: The buffer of the input tensor with index `i` of type T can
    // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
    auto input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
          
    // Copy input in proper precision 
    for (int i=0; i<224; i++){
      for (int j=0; j<224; j++){   
        cv::Vec3b pixel = inputs[seq % 2].at<cv::Vec3b>(i , j);
        *(input_tensor + i * 224*3 + j * 3) = ((uint8_t)pixel[0]);
        *(input_tensor + i * 224*3 + j * 3 + 1) = ((uint8_t)pixel[1]);
        *(input_tensor + i * 224*3 + j * 3 + 2) = ((uint8_t)pixel[2]);
      }
    }
    
    // Get start time
    clock_gettime(CLOCK_MONOTONIC, &begin);
    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Get end time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double temp = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    latency += temp;

    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
    auto output_tensor = interpreter->typed_output_tensor<uint8_t>(0);

    // Output print
    // for(int i=0; i<1000; ++i){
    //   printf("label : %d %d%\n", i, output_tensor[i]);
    // }
  }

  printf("Total elepsed time : %.6f sec\n", latency);
  printf("Average inference latency : %.6f sec\n", latency / inference_num);
  
  return 0;
}
