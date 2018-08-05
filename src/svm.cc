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

    return cv::ml::TrainData::create(features, 0, labels);
  }

  float train(const svm::factory::type& modelType, const data::factory::type& dataType){

    auto data = utils::data::factory::_new(dataType);
    auto model = svm::factory::_new(modelType);

    auto C = cv::ml::ParamGrid::create(10,100,10);

    auto gamma = cv::ml::ParamGrid::create(0.1,10,1);
    auto alpha = cv::ml::ParamGrid::create(0.1,10,1);
    auto beta = cv::ml::ParamGrid::create(0.1,10,1);
    auto nu = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU);
    auto p = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P);

    data->setTrainTestSplitRatio(0.2,true);

    //model->trainAuto(data,double(1.0),C,gamma,p,nu,alpha,beta,false);

    model->trainAuto(data);

    return model->calcError(data, true, cv::noArray());
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

  cv::Ptr<cv::ml::SVM> factory::_new(const svm::factory::type& type){

    auto algorithm = cv::ml::SVM::create();
    auto criteria = cv::TermCriteria();

    criteria.type = CV_TERMCRIT_EPS;
    criteria.epsilon = 1e-10;

    algorithm->setType(algorithm->C_SVC);
    algorithm->setTermCriteria(criteria);

    switch (type) {
      case linear:
        algorithm->setKernel(algorithm->LINEAR);
        break;
      case rbf:
        algorithm->setKernel(algorithm->RBF);
        break;
      case polymomial:
        algorithm->setDegree(3.0);
        algorithm->setKernel(algorithm->POLY);
        break;
      case sigmoid:
        algorithm->setKernel(algorithm->SIGMOID);
        break;
      case chi2:
        algorithm->setKernel(algorithm->CHI2);
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

   std::cout << utils::train(svm::factory::type::linear, utils::data::factory::type::circle) << "\n";

   std::cout << utils::train(svm::factory::type::linear, utils::data::factory::type::moon) << "\n";


   std::cout << utils::train(svm::factory::type::rbf, utils::data::factory::type::blob) << "\n";

   std::cout << utils::train(svm::factory::type::rbf, utils::data::factory::type::circle) << "\n";

   std::cout << utils::train(svm::factory::type::rbf, utils::data::factory::type::moon) << "\n";


   std::cout << utils::train(svm::factory::type::polymomial, utils::data::factory::type::blob) << "\n";

   std::cout << utils::train(svm::factory::type::polymomial, utils::data::factory::type::circle) << "\n";

   std::cout << utils::train(svm::factory::type::polymomial, utils::data::factory::type::moon) << "\n";


   std::cout << utils::train(svm::factory::type::sigmoid, utils::data::factory::type::blob) << "\n";

   std::cout << utils::train(svm::factory::type::sigmoid, utils::data::factory::type::circle) << "\n";

   std::cout << utils::train(svm::factory::type::sigmoid, utils::data::factory::type::moon) << "\n";


   std::cout << utils::train(svm::factory::type::chi2, utils::data::factory::type::blob) << "\n";

   std::cout << utils::train(svm::factory::type::chi2, utils::data::factory::type::circle) << "\n";

   std::cout << utils::train(svm::factory::type::chi2, utils::data::factory::type::moon) << "\n";

}
