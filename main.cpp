#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace cv::face;

#define COLOR Scalar(255, 200,0)
const string kPathToCV = "/Users/hwax0r/installation/OpenCV-master/share/opencv4/";
const string kPathToPrjct = "/Users/hwax0r/CLionProjects/ComputerVision/";

class moduleVideo{
private:
    // массив входящих кадров (буфер видео в 10 сек)
    vector<Mat> input_Frames;
    // массив обработанных кадров
    vector<Mat> processed_Frames;
public:
    string path;

    // TODO: что насчёт std::move()?
    moduleVideo(vector<Mat> vec_of_frames){
        this->input_Frames = vec_of_frames;
        this->processed_Frames.clear();
        checkInput();
    }

    // пустой конструктор для загрузки и обработки новой части кадров
    moduleVideo(){};
    void loadModel(vector<Mat> arrayOfFrames){
        this->input_Frames = arrayOfFrames;
        this->processed_Frames.clear();
        checkInput();
    }

    //
    void checkInput(){
        if (this->input_Frames.empty()){
            cout << "ERROR! Input is empty!" << endl;
        }
    }
    //
    void checkOutput(){
        if (!this->input_Frames.empty() && this->processed_Frames.empty()){
            cout << "ERROR! Input exists, but no faces found! processed_Frames is empty!" << endl;
        }
    }

    // Рисует линию на кадре из множества точек
    void drawPolyline
            (
                    Mat &im,                          // frame
                    const vector<Point2f> &landmarks, // точки на лице
                    const int start,                  // точка начала линии
                    const int end,                    // точка конца линии
                    bool is_closed = false            // замкнутая ли фигура
            )
    {
        vector <Point> points;
        for (int i = start; i <= end; i++)
        {
            points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
        }
        polylines(im, points, is_closed, COLOR, 2, 16);
    }

    // Рисует точки на лице и соединяет их
    // TODO: переделать в bool?
    void drawLandmarks
            (
                    Mat &im,                   // frame
                    vector<Point2f> &landmarks // точки на лице
            )
    {
        // Если точек достаточно для нахождения лица
        if (landmarks.size() == 68 &&
        abs((landmarks[16].x - landmarks[0].x)) > 0.2 * im.cols )
        {
            drawPolyline(im, landmarks, 0, 16);           // Jaw line
            drawPolyline(im, landmarks, 17, 21);          // Left eyebrow
            drawPolyline(im, landmarks, 22, 26);          // Right eyebrow
            drawPolyline(im, landmarks, 27, 30);          // Nose bridge
            drawPolyline(im, landmarks, 30, 35, true);    // Lower nose
            drawPolyline(im, landmarks, 36, 41, true);    // Left eye
            drawPolyline(im, landmarks, 42, 47, true);    // Right Eye
            drawPolyline(im, landmarks, 48, 59, true);    // Outer lip
            drawPolyline(im, landmarks, 60, 67, true);    // Inner lip
        }
//        else // только если распознана лишь часть точек или лицо маленькое.
//        {
//            for(int i = 0; i < landmarks.size(); i++)
//            {
//                circle(im,landmarks[i],3, COLOR, FILLED);
//            }
//        }
    }

    // поиск лиц
    void faceDetection(){
        vector<Mat> result;
        // загрузка модели определения лица в анфас
        CascadeClassifier face_detector;
        path = kPathToCV + "haarcascades/haarcascade_frontalface_alt2.xml";
        face_detector.load(path);

        // создание обработчика изображения
        Ptr<FacemarkLBF> facemark = FacemarkLBF::create();
        // загрузка обученной модели  iBUG 300-W dataset
        path = kPathToPrjct + "lbfmodel.yaml";
        facemark->loadModel(path);

        Mat gray, frame;

        // поиск лица на каждом кадре
        for (int i = 0; i < this->input_Frames.size() - 1; ++i)
        {
            // получаем кадр из массива
            frame = this->input_Frames[i];

            // Find face
            vector<Rect> faces;

            // преобразуем кадр к серому и соxраняем (frame -> gray)
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // поиск лица на сером кадре по "признакам" Хаара
            face_detector.detectMultiScale(gray, faces);

            /*
             * переменная для точек на лице
             * вектор векторов нужен для нахождения всех лиц на кадре
             * (лиц может быть больше одного)
             */
            vector<vector<Point2f>> landmarks;

            // Run landmark detector
            bool success = facemark->fit(frame, faces, landmarks);

            // если лицо распознано, отрисовываем точки на лице
            if (success)
            {
                for (int i = 0; i < landmarks.size(); i++)
                {
                    drawLandmarks(frame, landmarks[i]);
                }
                result.push_back(frame);
            }
        }
        this->processed_Frames = result;
    }

    // вывод результата обработки в окно
    void showInWindow(){
        if (this->processed_Frames.empty()){
            cout << " ERROR! No frames with face on them. " << endl;
            return;
        }
        cout << "Results (Window mode) started." << endl;
        for (const Mat& frame:this->processed_Frames){
            imshow("processed", frame);
            if (waitKey(1) == 27) break;
        }
        cout << "Results (Window mode) end of function." << endl;
    }

    // вывод результата обработки в виде .img файлов
    void saveInFolder(){
        if (this->processed_Frames.empty()){
            cout << "ERROR! No frames with face on them." << endl;
            return;
        }
        cout << "Results (.jpg saving in a folder) started" << endl;
        for (int i = 0; i < this->processed_Frames.size(); ++i){
            // путь к папке из корня + images/processed_%i.jpg
            imwrite(format("/Users/hwax0r/CLionProjects/ComputerVision/images/processed_%i.jpg", i),
                    this->processed_Frames[i]);
        }
        cout << "Results (.jpg saved in a folder)." << endl;
    }

    // проверка на ошибки

};



int main(int argc, char** argv)
{
    vector<Mat> test;
    VideoCapture movie("/Users/hwax0r/CLionProjects/ComputerVision/testMovie.mov");

    bool end = false;
    while (!end)
    {
        Mat frame;
        end = !(movie.read(frame));
        test.push_back(frame);
    }

//    moduleVideo newDetection(test);
    moduleVideo newDetection;
    newDetection.loadModel(test);
    newDetection.faceDetection();
//    newDetection.showInWindow();
//    newDetection.saveInFolder();

    return 0;
}
