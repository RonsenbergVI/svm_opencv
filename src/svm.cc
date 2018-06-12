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


namespace svm{

cv::Ptr<cv::ml::StatModel> factory::_new(const svm::factory::type & type){

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
    default:
      throw "invalid type.";
  }

  return algorithm;
}

};

int main(){

  
}
