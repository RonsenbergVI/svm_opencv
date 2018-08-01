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

#include "svm.hh"

std::string utils::data::factory::path = std::string("");

namespace utils{

  cv::Ptr<cv::ml::TrainData> read(std::string path){

    cv::Mat feature_row = cv::Mat::ones(1, 2, CV_32F);

    cv::Mat label_row = cv::Mat::ones(1, 1, CV_32S);

    cv::Mat features, labels;

    std::string str;

    std::cout << path << "\n";

    std::ifstream file(path);

    float x,y,z;

    while(getline(file, str, '\n')){

        std::stringstream ss(str);

        ss >> x >> y >> z;

        feature_row.at<float>(0, 0) = x;

        feature_row.at<float>(0, 1) = y;

        label_row.at<float>(0, 0) = z;

        features.push_back(feature_row);

        labels.push_back(label_row);
    }

    auto result = cv::ml::TrainData::create(features, 0, labels);

    return result;
  }

  float train(const svm::factory::type& modelType, const data::factory::type& dataType){

    auto data = utils::data::factory::_new(dataType);
    auto model = svm::factory::_new(modelType);

    auto y_ = cv::OutputArray(cv::Mat());

    y_.clear();

    data->setTrainTestSplitRatio(0.2,true);

    model->train(data);

    auto error = model->calcError(data, true, y_);

    return error;
  }

  namespace data{

    std::string factory::text(const factory::type& type){
      switch(type){
      case blob:
        return "blob";
      case circle:
        return "circle";
      case moon:
        return "moon" ;
      default:
        return "unknown";
      }
    }

    cv::Ptr<cv::ml::TrainData> factory::_new(const data::factory::type& type){
      return read(utils::data::factory::path+text(type)+".txt");
    }
  }
}

namespace svm{

  cv::Ptr<cv::ml::StatModel> factory::_new(const svm::factory::type& type){

    auto algorithm = cv::ml::SVM::create();
    auto criteria = cv::TermCriteria();

    criteria.type = CV_TERMCRIT_EPS;
    criteria.epsilon = 1e-10;

    switch (type) {
      case linear:
        algorithm->setC(100);
        algorithm->setKernel(algorithm->LINEAR);
        algorithm->setTermCriteria(criteria);
        algorithm->setType(algorithm->C_SVC);
        break;
      case rbf:
        algorithm->setC(100);
        algorithm->setGamma(0.1);
        algorithm->setCoef0(0.3);
        algorithm->setTermCriteria(criteria);
        algorithm->setKernel(algorithm->RBF);
        algorithm->setType(algorithm->C_SVC);
        break;
      case polymomial:
        algorithm->setC(100);
        algorithm->setGamma(0.1);
        algorithm->setCoef0(0.3);
        algorithm->setTermCriteria(criteria);
        algorithm->setKernel(algorithm->POLY);
        algorithm->setType(algorithm->C_SVC);
        break;
      case sigmoid:
        algorithm->setC(100);
        algorithm->setGamma(0.1);
        algorithm->setCoef0(0.3);
        algorithm->setTermCriteria(criteria);
        algorithm->setKernel(algorithm->SIGMOID);
        algorithm->setType(algorithm->C_SVC);
        break;
      case chi2:
        algorithm->setC(100);
        algorithm->setGamma(0.1);
        algorithm->setCoef0(0.3);
        algorithm->setTermCriteria(criteria);
        algorithm->setKernel(algorithm->CHI2);
        algorithm->setType(algorithm->C_SVC);
        break;
      default:
        throw "invalid type";
    }

    return algorithm;

  }

};

int main(int argc, const char * argv[]) {

   utils::data::factory::path = std::string(argv[1]);

   std::cout << utils::train(svm::factory::type::linear, utils::data::factory::type::blob) << "\n";

}
