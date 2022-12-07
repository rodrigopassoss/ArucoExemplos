// Primeiro código exemplo da biblioteca ARUCO, 2022
/*
 * DESCRIÇÃO:
 *
 * Este código Usa o dicionário de marcadores "aruco"
 * disponível na openCV para gerar um marcador, que posteriormente
 * pode ser impresso para a aplicação desejada.
 *
 * DETALHAMENTO:
 *
 * Os marcadores são compostos por um quatrado com bora preta, e uma matriz binária em
 * em seu interior (preto-branco), cuja sua configuração define o "id" do marcador. O "id",
 * é simplesmento o indice do marcador no discionário
 *
 */


#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv4/opencv2/aruco.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>


using namespace std;
using namespace cv;


const float calibrationSquareDimension = 0.02405f; //medida do quadrado em metros
const float arucoSquareDimensions = 0.0536f;       //tamanho medido dos quadados dos marcadores aruco
const Size chessBoardDimensions = Size(6,9);


void createArukoMakers()
{
    Mat outputMarker;  //Declara a matriz que armazenará a imagem do marcador

    // Cria o dicionário de marcadoes: 250 marcadores de Matrizes 6x6
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_50);

    for(int i = 0;i < 50; i++)
    {
        aruco::drawMarker(markerDictionary,i,200,outputMarker,1);
        ostringstream convert;
        string imageName = "6x6Marker_";
        convert << imageName << i << ".png";
        imwrite(convert.str(),outputMarker);
    }
}

void createKnowBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
    for(int i = 0; i < boardSize.height ; i++)
    {
        for(int j = 0; j < boardSize.width ; j++)
        {
            corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength,0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
    for(vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
    {
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(9,6),pointBuf, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if(found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if(showResults)
        {
            drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
            imshow("Looking for Corners", *iter);
            waitKey(0);
        }

    }

}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    vector<vector<Point2f>> checkboardImageSpacePoints;
    getChessboardCorners(calibrationImages, checkboardImageSpacePoints, false);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);
    createKnowBoardPosition(boardSize,squareEdgeLength,worldSpaceCornerPoints[0]);

    worldSpaceCornerPoints.resize(checkboardImageSpacePoints.size(),worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8,1,CV_64F);
    calibrateCamera(worldSpaceCornerPoints,checkboardImageSpacePoints,boardSize,cameraMatrix,distanceCoefficients, rVectors, tVectors);

}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
    ofstream outStream(name);

    if(outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t cols = cameraMatrix.cols;

        outStream << rows << endl;
        outStream << cols << endl;


        for(int r=0; r<rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                double value = cameraMatrix.at<double>(r,c);

                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        cols = distanceCoefficients.cols;

        outStream << rows << endl;
        outStream << cols << endl;

        for(int r=0; r<rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                double value = distanceCoefficients.at<double>(r,c);

                outStream << value << endl;
            }

        }

        outStream.close();
        return true;

     }

      return false;

}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if(inStream)
    {
        uint16_t rows;
        uint16_t cols;

        inStream >> rows;
        inStream >> cols;

        cameraMatrix = Mat(Size(cols,rows), CV_64F);

        for(int r=0; r<rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r,c) = read;

                cout << cameraMatrix.at<double>(r,c) << "\n";
            }
        }

        inStream >> rows;
        inStream >> cols;

        distanceCoefficients = Mat::zeros(cols,rows, CV_64F);

        for(int r=0; r<rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                double read = 0.0f;
                inStream >> read;
                distanceCoefficients.at<double>(r,c) = read;

                cout << distanceCoefficients.at<double>(r,c) << "\n";
            }
        }
        inStream.close();

        return true;

    }

    return false;
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimensions)
{
    Mat frame;

    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    aruco::DetectorParameters params;
    Ptr< aruco::Dictionary > markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_50);

    VideoCapture vid(0);

    if(!vid.isOpened())
    {
         return -1;
    }

    int framesPerSecond = 20;

    namedWindow("WebCam", WINDOW_AUTOSIZE);

    vector<Vec3d> rotationVectors, translationVectors;

    while(true)
    {

        if(!vid.read(frame))
            break;

        aruco::detectMarkers(frame,markerDictionary,markerCorners,markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners,arucoSquareDimensions, cameraMatrix, distanceCoefficients,rotationVectors,translationVectors);

        for(int i = 0; i < (int)markerIds.size(); i++)
        {
            aruco::drawAxis(frame, cameraMatrix,distanceCoefficients,rotationVectors[i],translationVectors[i],0.1f);
        }

        imshow("WebCam",frame);

        if(waitKey(1000/framesPerSecond) >= 0) break;



    }

    return 1;


}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
    Mat frame;
    Mat drawToFrame;

    vector<Mat> savedImages;

    vector<vector<Point2f>> makerCorners, rejectedCandidates;

    VideoCapture vid(0);

    if(!vid.isOpened())
    {
         return;
    }

    int framesPerSecond = 20;

    namedWindow("WebCam", WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
            break;

        vector<Vec2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, chessBoardDimensions,foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessBoardDimensions, foundPoints, found);


        if(found)
            imshow("WebCam",drawToFrame);
        else
            imshow("WebCam",frame);

        int character = waitKey(1000/framesPerSecond);
        cout << character << endl;


        switch (character)
        {
            case ' ':
                 //saving image
                 if(found)
                 {
                     Mat tmp;
                     frame.copyTo(tmp);
                     savedImages.push_back(tmp);
                 }
                 cout << "Saving Images" << endl;
                  break;
            case 13:
                  //start calibration
                  if(savedImages.size() > 15)
                  {
                      cameraCalibration(savedImages,chessBoardDimensions,calibrationSquareDimension, cameraMatrix,distanceCoefficients);
                      saveCameraCalibration("wCam_calibration_data",cameraMatrix,distanceCoefficients);
                  }
                  break;
            case 27:
                  // exit
                  break;

        }

    }

}

void markerDetectionProcess()
{
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    int framesPerSecond = 20;

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        if (ids.size() > 0)
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(1000/framesPerSecond);
        if (key == 13)
            break;
    }

}

int main()
{
  /*
    EXEMPLO DE CRIAÇÃO DE UM MARCADOR:

    Mat markerImage;  //Declara a matriz que armazenará a imagem do marcador

    // Cria o dicionário de marcadoes: 250 marcadores de Matrizes 6x6
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    // desenha o marcador de id = 23 em uma imagem 200x200 pixels
    aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
    //Cria a imgem png
    imwrite("marker23.png", markerImage);
    //Mostra imagem
    imshow("test",markerImage);
    waitKey(0); // Wait for a keystroke in the window

  */


    Mat cameraMatrix = Mat::eye(3,3,CV_64F);
    Mat distanceCoefficients;

    //cameraCalibrationProcess(cameraMatrix,distanceCoefficients);

    loadCameraCalibration("wCam_calibration_data",cameraMatrix,distanceCoefficients);
    startWebcamMonitoring(cameraMatrix,distanceCoefficients,arucoSquareDimensions);
    //markerDetectionProcess();




    return 0;
}
