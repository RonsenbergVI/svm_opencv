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

namespace utils{

  cv::Ptr<cv::ml::TrainData> read(std::string path){

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

  void train(svm::factory::data modelType, utils::data::factory::type dataType){

    auto model = _new(modelType);
    auto data = _new(dataType);

    auto y_ = cv::OutputArray(cv::Mat());

    y_.clear();

    trainData->setTrainTestSplitRatio(0.2,true);

    model->train(data);

    auto error = model->calcError(trainData, true, y_);

    auto filename = path + "_linear.txt";

    cv::Mat features = cv::Mat::ones(1, 2, CV_32F);

    feature_row = data.second;

    cv::Mat labels = cv::Mat();

    model.second->predict(feature_row,y_);

    std::ofstream output(filename);

    output << "x1" << "," << "x2" << "," << "y" << "\n";

    for(int i = 0; i < feature_row.rows; i++){

        auto input = cv::InputArray(feature_row.row(i));

        output << feature_row.at<float>(i,0) << "," << feature_row.at<float>(i,1)  << "," <<  y_predict.getMatRef().at<float>(i,0) << "\n";
    }
  }

  namespace data{

    std::string factory::text(data::factory::type type){
      switch()
      {
      case blob:
        return "blob";
      case circle:
        return "circle";
      case Enum::Apple:
        return "moon" ;
      default:
        return "unknown";
      }
    }

    cv::Ptr<cv::ml::TrainData> factory::_new(const svm::factory::type& type){
      return readTextFile(path+text(type));
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

};

int main(int argc, const char * argv[]) {

   utils::data::factory::path = std::string(argv[1]);

   //auto data = utils::data::factory::_new(utils::data::factory::type::blob);
   //auto algo = utils::svm::factory::_new(utils::svm::factory::type::linear);

   train(svm::factory::type::linear,utils::data::factory::type::blob);

}
