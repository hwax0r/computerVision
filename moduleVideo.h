//
// Created by David Sergeev on 30.07.2020.
//
#ifndef COMPUTERVISION_MODULEVIDEO_H
#define COMPUTERVISION_MODULEVIDEO_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <map>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace chrono;
using namespace chrono_literals;

#define COLOR Scalar(255, 200,0)
const string kPathToCV = "/Users/hwax0r/installation/OpenCV-master/share/opencv4/";
const string kPathToPrjct = "/Users/hwax0r/CLionProjects/ComputerVision/";

struct Timer{
    chrono::time_point<chrono::steady_clock> start, end;
    chrono::duration<float> duration;

    Timer(){
        start = chrono::high_resolution_clock::now();
    }

    ~Timer(){
        end = chrono::high_resolution_clock::now();
        duration = end - start;

        //float ms = duration.count() * 1000.0f; // in ms
        float sec = ceil(duration.count()); // in sec
        // Rounding value(makes everything from 0.0...1 to 0.99...9 equals 1)

        cout << "Function takes: " << sec << "_sec\n";
    }
};

class moduleVideo{
private:
    // массив входящих кадров (буфер видео в 10 сек)
    vector<Mat> input_Frames;
    // массив обработанных кадров
    vector<Mat> processed_Frames;

    //нужно для единоразовой подгрузки нейронки и модели лица
    Ptr<FacemarkLBF> facemark;
    CascadeClassifier face_detector;

public:
    string path;

    // TODO: что насчёт std::move()?

    moduleVideo(vector<Mat> vec_of_frames){
        cout << "moduleVideo started its work\n";
        this->input_Frames = vec_of_frames;
        this->processed_Frames.clear();
        checkInput();
    }

    // TODO: что насчёт std::move()?

    // пустой конструктор для загрузки и обработки новой части кадров
    moduleVideo(){};
    void loadModel(vector<Mat> arrayOfFrames){
        cout << "moduleVideo started its work\n";
        Timer timer;
        this->input_Frames = arrayOfFrames;
        this->processed_Frames.clear();
        checkInput();
    }

    ~moduleVideo(){
        this->input_Frames.clear();
        this->processed_Frames.clear();
        cout << "moduleVideo not working anymore\n";
    }

    // проверка на ошибки
    // TODO: void -> int; расписать номера ошибок для return

    void checkInput(){
        if (this->input_Frames.empty()){
            cout << "ERROR! Input is empty!" << endl;
        }
    }

    // TODO: void -> int; расписать номера ошибок для return

    void checkOutput(){
        if (!this->input_Frames.empty() && this->processed_Frames.empty()){
            cout << "ERROR! Input exists, but no faces found! processed_Frames is empty!" << endl;
        }
        checkInput();
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
    // TODO: переделать в bool/не переделывать в bool

    void drawLandmarks
            (
                    Mat &im,                   // frame
                    vector<Point2f> &landmarks // точки на лице
            )
    {
        // Если точек достаточно для нахождения лица и ширина лица больше 20% ширины кадра
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
            //return true;
        }; //else return false;
//        else // только если распознана лишь часть точек или лицо маленькое.
//        {
//            for(int i = 0; i < landmarks.size(); i++)
//            {
//                circle(im,landmarks[i],3, COLOR, FILLED);
//            }
//        }
    }

    // поиск лиц

    // TODO: разделить определение лиц / загрузку модели лица и датасета
    //  загрузка моделей занимает очень много времени

    void loadClassifierAndFacemarks(){
        cout << "loadClassifierAndFacemarks(): ";
        Timer timer;
        // загрузка модели определения лица в анфас
        path = kPathToCV + "haarcascades/haarcascade_frontalface_alt2.xml";
        this->face_detector.load(path);

        // создание обработчика изображения
        this->facemark = FacemarkLBF::create();
        // загрузка обученной модели  iBUG 300-W dataset
        path = kPathToPrjct + "lbfmodel.yaml";
        this->facemark->loadModel(path);
    }

    void faceDetection(){
        cout << "faceDetection for " << this->input_Frames.size() << " frames\n";
        Timer timer;

        vector<Mat> result;
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
            this->face_detector.detectMultiScale(gray, faces);

            /*
             * переменная для точек на лице
             * вектор векторов нужен для нахождения всех лиц на кадре
             * (лиц может быть больше одного)
             */
            vector<vector<Point2f>> landmarks;

            // Run landmark detector
            bool success = this->facemark->fit(frame, faces, landmarks);

            // если лицо распознано, отрисовываем точки на лице
            if (success)
            {
                // TODO:
                //      BUG FOUND!
                //      Функция то ли не рисует на некоторых кадрах,
                //      то ли сохраняет кадры, где лицо не определено.
                //      (В выводе были кадры без рисунка) (Как?)

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

    // TODO: вызвать функцию обработки ошибок и (прерывание?)

    void showInWindow(){
        if (this->processed_Frames.empty()){
            cout << "ERROR! No frames with face on them. " << endl;
            return;
        }
        cout << "Results (Window mode) started." << endl;
        for (const Mat& frame:this->processed_Frames){
            imshow("processed", frame);
            if (waitKey(3) == 27) break;
        }
        cout << "Results (Window mode) end of function." << endl;
    }

    // вывод результата обработки в виде .img файлов
    // TODO: вызвать функцию обработки ошибок и (прерывание?)

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

    //TODO:
    // Написать функцию проверки на резкость изображения с помощью выборки.
    // Найти середину линии мажду точками landmarks с левой/правой части подбородка.
    // Создать прямоугольник с центром в этой точке, один краем со внешней стороны лица,
    // другим краем, смотрящим в сторону носа/центра лица.

    void algorithm(Mat& frame, vector<Point2f> &landmarks){
    }

};


#endif //COMPUTERVISION_MODULEVIDEO_H
