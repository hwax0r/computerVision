#include <vector>
#include <unistd.h> // sleep()
#include "moduleVideo.h"

vector<Mat> testInput(){
    Timer timer;
    vector<Mat> testFrames;
    VideoCapture movie("/Users/hwax0r/CLionProjects/ComputerVision/testMovie.mov");

    bool end = false;
    while (!end)
    {
        Mat frame;
        end = !(movie.read(frame));
        testFrames.push_back(frame);
    }

    return testFrames;
}

int main()
{
    vector<Mat> test = testInput();

    cout << "Всё ок" << endl;
    moduleVideo newDetection;
    newDetection.loadClassifierAndFacemarks();
    newDetection.loadModel(test);
    newDetection.faceDetection();
//    newDetection.saveInFolder();

    cout << "ошибки" << endl;
    moduleVideo kekw;
    kekw.loadClassifierAndFacemarks();
    kekw.checkInput();
    vector<Mat> kek(1);
    kekw.loadModel(kek);
    kekw.checkOutput();
    kekw.showInWindow();
    kekw.saveInFolder();

    return 0;
}
