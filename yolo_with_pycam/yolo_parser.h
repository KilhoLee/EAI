#include <string>
#include <vector>
#include <complex>
#include <algorithm>
#include <iostream>
#include "tensorflow/lite/c/common.h"
#define IMG_size 416
#define Cls_thresh 0.2
// #define SOFTMAX

namespace yolo{
    class YOLO_Parser{
      public:
        YOLO_Parser() {};
        ~YOLO_Parser() {};
        static std::vector<std::vector<float>> real_bbox_cls_vector; 
        static std::vector<int> real_bbox_cls_index_vector;
        static std::vector<std::vector<int>> real_bbox_loc_vector;
        std::vector<int> get_cls_index(std::vector<std::vector<float>>& real_bbox_cls_vector){
            float max=0;
            int max_index = -1;
            int index = 0;
            for (auto i : real_bbox_cls_vector) { 
              index = 0;
            		for (auto j : i) { 
                if (j > max){
                  max = j;
                  max_index = index;
                }
                index+=1;
            		}
              real_bbox_cls_index_vector.push_back(max_index);
              max = 0;
              max_index = -1;
            	}
            return real_bbox_cls_index_vector;
        };
        void make_real_bbox_cls_vector(TfLiteTensor* cls_tensor, std::vector<int>& real_bbox_index_vector, 
                                        std::vector<std::vector<float>>& real_bbox_cls_vector){
            TfLiteTensor* output_tensor = cls_tensor;  
            const float* output_data = (float*)output_tensor->data.data;
            const int num_raw_bboxes = output_tensor->dims->data[1]; 
            std::vector<float> classifications;
            for (int i = 0; i < num_raw_bboxes; ++i) {
              for (int j = 0; j < 80; ++j) {
                  classifications.push_back(output_data[i*80 + j]);  
                 }
            }
            std::vector<float> raw_vector;
            for (int i = 0; i < num_raw_bboxes; ++i) {
              bool is_survived = false;
              for (int j = 0; j < 80; ++j) {
                raw_vector.push_back(classifications[i * 80 + j]); 
              }
              #ifdef SOFTMAX
              // SOFTMAX(raw_vector); // Not use Softmax currently
              #endif
              for (int k = 0; k < 80; ++k) {
                if (raw_vector[k] > Cls_thresh){
                  is_survived = true;
                }
              }
              if(is_survived){
                real_bbox_index_vector.push_back(i); 
                real_bbox_cls_vector.push_back(raw_vector);
              }
              raw_vector.clear();
              }
              classifications.clear();
              printf("\033[0;32mBefore NMS : \033[0m");
              std::cout << " Number of bounding boxes before NMS : " << real_bbox_index_vector.size() << std::endl;
        };
        void make_real_bbox_loc_vector(TfLiteTensor* loc_tensor, std::vector<int>& real_bbox_index_vector,
                                        std::vector<std::vector<int>>& real_bbox_loc_vector){
            TfLiteTensor* output_tensor = loc_tensor;
            auto input_pointer = (float *)output_tensor->data.data;
            const float* output_data = (float*)output_tensor->data.data; 
            const int num_raw_bboxes = output_tensor->dims->data[1]; 
            const int num_columns = output_tensor->dims->data[2]; 
            std::vector<float> boxes;
            for (int i = 0; i < num_raw_bboxes; ++i) {
                 for (int j = 0; j < num_columns; ++j) {
                    boxes.push_back(output_data[i * 4 + j]);  
                 }
            }
            for (int i = 0; i < num_raw_bboxes; ++i) {
                std::vector<int>tmp;
                for(int j=0 ; j < real_bbox_index_vector.size(); j++){
                    if(i == real_bbox_index_vector[j]) {
                      float first = boxes[i * 4];      
                      float second = boxes[i * 4 + 1]; 
                      float third = boxes[i * 4 + 2]; 
                      float fourth = boxes[i* 4 + 3];   
                      int left = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
                      (IMG_size), first - third/2)));
                      int top = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
                      (IMG_size), second - fourth/2)));
                      int right = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
                      (IMG_size), first + third/2)));
                      int bottom = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
                      (IMG_size), second + fourth/2)));
                      tmp.push_back(left);
                      tmp.push_back(top);
                      tmp.push_back(right);
                      tmp.push_back(bottom);
                      real_bbox_loc_vector.push_back(tmp);
                      break;
                    }
                }
                tmp.clear();
            }
        };
        void SOFTMAX(std::vector<float>& row){
            const float threshold = 0.999999; 
            float maxElement = *std::max_element(row.begin(), row.end());
            float sum = 0.0;
            const float scalingFactor = 20.0; //20
            for (auto& i : row)
                sum += std::exp(scalingFactor * (i - maxElement));
            for (int i = 0; i < row.size(); ++i) {
                row[i] = std::exp(scalingFactor * (row[i] - maxElement)) / sum;
                if (row[i] > threshold)
                    row[i] = threshold; 
            }
        };
        struct BoundingBox {
          float left, top, right, bottom;
          float score;
          int class_id;
        };
        static std::vector<YOLO_Parser::BoundingBox> result_boxes;
        static bool CompareBoxesByScore(const BoundingBox& box1, const BoundingBox& box2){
            return box1.score > box2.score; };   
        float CalculateIoU(const BoundingBox& box1, const BoundingBox& box2){
            float x1 = std::max(box1.left, box2.left);
            float y1 = std::max(box1.top, box2.top);
            float x2 = std::min(box1.right, box2.right);
            float y2 = std::min(box1.bottom, box2.bottom);
            float area_box1 = (box1.right - box1.left) * (box1.bottom - box1.top);
            float area_box2 = (box2.right - box2.left) * (box2.bottom - box2.top);
            float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float union_area = area_box1 + area_box2 - intersection_area;

            if (union_area > 0.0f) {
                return intersection_area / union_area;
            } else {
                return 0.0f; 
            }
        };
        void NonMaximumSuppression(std::vector<BoundingBox>& boxes, float iou_threshold){
            std::sort(boxes.begin(), boxes.end(), CompareBoxesByScore);
            std::vector<BoundingBox> selected_boxes;
            while (!boxes.empty()) {
                BoundingBox current_box = boxes[0];
                selected_boxes.push_back(current_box);
                boxes.erase(boxes.begin());
                for (auto it = boxes.begin(); it != boxes.end();) {
                    float iou = CalculateIoU(current_box, *it);
                    if (iou > iou_threshold) {
                        it = boxes.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            boxes = selected_boxes;
        };
        void PerformNMSUsingResults(
        const std::vector<int>& real_bbox_index_vector,
        const std::vector<std::vector<float>>& real_bbox_cls_vector,
        const std::vector<std::vector<int>>& real_bbox_loc_vector,
        float iou_threshold, const std::vector<int> real_bbox_cls_index_vector) {
            std::vector<BoundingBox> bounding_boxes;
            for (size_t i = 0; i < real_bbox_index_vector.size(); ++i) {
                BoundingBox box;
                box.left = static_cast<float>(real_bbox_loc_vector[i][0]);
                box.top = static_cast<float>(real_bbox_loc_vector[i][1]);
                box.right = static_cast<float>(real_bbox_loc_vector[i][2]);
                box.bottom = static_cast<float>(real_bbox_loc_vector[i][3]);
                box.score = static_cast<float>(real_bbox_cls_vector[i][real_bbox_cls_index_vector[i]]); 
                box.class_id = real_bbox_cls_index_vector[i];
                bounding_boxes.push_back(box);
            }
            NonMaximumSuppression(bounding_boxes, iou_threshold);
            printf("\033[0;32mAfter NMS : \033[0m");
            printf("Number of bounding boxes after NMS: %zu\n",bounding_boxes.size());
            result_boxes = bounding_boxes;
            bounding_boxes.clear();
      };
      void visualize_with_labels(cv::Mat& image, const std::vector<BoundingBox>& bboxes, std::map<int, std::string>& labelDict) {
        for (const BoundingBox& bbox : bboxes) {
            int x1 = bbox.left;
            int y1 = bbox.top;
            int x2 = bbox.right;
            int y2 = bbox.bottom;
            cv::RNG rng(bbox.class_id);
            cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            int label_x = x1;
            int label_y = y1 - 20;

            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
            std::string object_name = labelDict[bbox.class_id];
            float confidence_score = bbox.score;
            std::string label = object_name + ": " + std::to_string(confidence_score);
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            cv::rectangle(image, cv::Point(x1, label_y - text_size.height), cv::Point(x1 + text_size.width, label_y + 5), color, -1);
            cv::putText(image, label, cv::Point(x1, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }
    };
    };
}