SOURCE="${BASH_SOURCE[0]}"

DIR="$( cd -P "$( dirname "$SOURCE")" && pwd )/extracts/"

g++ src/svm.cc -lopencv_core -lopencv_ml -lopencv_imgproc -o src/opencv.out

src/opencv.out $DIR
