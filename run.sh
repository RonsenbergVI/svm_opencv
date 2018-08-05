#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"

DIR="$( cd -P "$( dirname "$SOURCE")" && pwd )/data/"

g++ -std=c++17 src/svm.cc -lopencv_core -lopencv_ml -lopencv_imgproc -o src/svm.out

src/svm.out $DIR
