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


namespace svm{
  class factory{
    public:

    enum type {
      linear,
      polymomial,
      rbf,
      sigmoid,
      chi2
    };

    static cv::Ptr<cv::ml::StatModel> _new(const factory::type&);

  };
}

namespace utils{

  namespace data{
    class factory{
    public:
      enum type{
        blob,
        circle,
        moon
      };

      static std::string text(const data::factory::type&);
      static cv::Ptr<cv::ml::TrainData> _new(const data::factory::type&);
      static std::string path;

    };
  }

  cv::Ptr<cv::ml::TrainData> read(std::string);
  double train(const svm::factory::type& ,const data::factory::type&);
}
