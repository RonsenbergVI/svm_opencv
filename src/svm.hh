// MIT License
//
// Copyright (c) 2018 Rene Jean Corneille
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>


namespace utils{

  cv::Ptr<cv::ml::TrainData> readTextFile(std::string path){

    std::pair<std::string, cv::Mat> features_pair,labels_pair;

    cv::Mat feature_row = cv::Mat::ones(1, 2, CV_32F);

    cv::Mat label_row = cv::Mat::ones(1, 1, CV_32S);

    cv::Mat features, labels;

    std::string str;

    std::ifstream file(path);

    float x,y,z;

    while(getline(file, str, '\n')){

        std::stringstream ss(str);

        ss >> x >> y >> z;

        std::cout << x << "\n";

        feature_row.at<float>(0, 0) = x;

        feature_row.at<float>(0, 1) = y;

        label_row.at<float>(0, 0) = z;

        features.push_back(feature_row);

        labels.push_back(label_row);
    }

    cv::Ptr<cv::ml::TrainData> result = cv::ml::TrainData::create(features, 0, labels);

    return result;
  }

  namespace data{
    class factory{
    public:
      enum type{
        blob,
        circle,
        moon
      };

      static cv::Ptr<cv::ml::TrainData> _new(const utils::data::factory::type&);

    private:
      static std::string path;

    };
  };
};

namespace svm{

  class factory{
    public:

    enum type {
      linear,
      rbf,
      sigmoid
    };

    static cv::Ptr<cv::ml::StatModel> _new(const svm::factory::type&);
  };
};
