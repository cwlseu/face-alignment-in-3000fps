//
//  AlignmentClip wenlongcao
//
//  Created by wenlong on 8/20/17.
//  Copyright (c) 2017 wenlong. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;

string modelPath = "./model/";
string dataPath = "./Datasets/";
string cascadeName = "haarcascade_frontalface_alt.xml";

void InitializeGlobalParam();
void PrintHelp();


int main( int argc, const char** argv )
{
    // main process
    if (argc == 1) {
        PrintHelp();
    }
    else if (argc == 2) {
        ReadGlobalParamFromFile(modelPath+"LBF.model");
        FaceDetectionAndAlignment(argv[1]);
    }
    return 0;
}

// set the parameters when training models.
void InitializeGlobalParam() {
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = 68;
    global_params.initial_num = 5;

    global_params.max_numstage = 7;
    double m_max_radio_radius[10] = {0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05};
    double m_max_numfeats[10] = {500, 500, 500, 300, 300, 200, 200, 200, 100, 100};
    for (int i = 0; i < 10; i++) {
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i = 0; i < 10; i++) {
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
    global_params.max_numthreshs = 500;
}

void ReadGlobalParamFromFile(string path) {
    cout << "Loading GlobalParam..." << endl;
    ifstream fin;
    fin.open(path);
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;

    for (int i = 0; i < global_params.max_numstage; i++) {
        fin >> global_params.max_radio_radius[i];
    }

    for (int i = 0; i < global_params.max_numstage; i++) {
        fin >> global_params.max_numfeats[i];
    }
    std::cout << "Loading GlobalParam end" << std::endl;
    fin.close();
}
void PrintHelp() {
    std::cout << "Useage:" << std::endl;
    std::cout << "test model on a pic:     LBF xx.jpg" << std::endl;
    std::cout << std::endl;
}

int save_count = 0;
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    LBFRegressor& regressor,
                    double scale, bool tryflip ) {
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0, 0, 255),
                                      CV_RGB(0, 128, 255),
                                      CV_RGB(0, 255, 255),
                                      CV_RGB(0, 255, 0),
                                      CV_RGB(255, 128, 0),
                                      CV_RGB(255, 255, 0),
                                      CV_RGB(255, 0, 0),
                                      CV_RGB(255, 0, 255)
                                    } ;
    Mat gray, smallImg( cvRound (img.rows / scale), cvRound(img.cols / scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    // --Detection
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
                              1.1, 2, 0
                              //|CV_HAAR_FIND_BIGGEST_OBJECT
                              | CV_HAAR_DO_ROUGH_SEARCH
                              | CV_HAAR_SCALE_IMAGE
                              ,
                              Size(30, 30) );
    if ( tryflip ) {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                  1.1, 2, 0
                                  //|CV_HAAR_FIND_BIGGEST_OBJECT
                                  //|CV_HAAR_DO_ROUGH_SEARCH
                                  | CV_HAAR_SCALE_IMAGE
                                  ,
                                  Size(30, 30) );
        for ( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }

    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t / ((double)cvGetTickFrequency() * 1000.) );

    // --Alignment
    t = (double)cvGetTickCount();
    for ( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Point center;
        Scalar color = colors[i % 8];
        BoundingBox boundingbox;

        boundingbox.start_x = r->x * scale;
        boundingbox.start_y = r->y * scale;
        boundingbox.width   = (r->width - 1) * scale;
        boundingbox.height  = (r->height - 1) * scale;
        boundingbox.centroid_x = boundingbox.start_x + boundingbox.width / 2.0;
        boundingbox.centroid_y = boundingbox.start_y + boundingbox.height / 2.0;

        t = (double)cvGetTickCount();
        Mat_<double> current_shape = regressor.Predict(gray, boundingbox, 1);
        t = (double)cvGetTickCount() - t;
        printf( "alignment time = %g ms\n", t / ((double)cvGetTickFrequency() * 1000.) );
        // draw bounding box
        rectangle(img, cvPoint(boundingbox.start_x, boundingbox.start_y),
                  cvPoint(boundingbox.start_x + boundingbox.width, boundingbox.start_y + boundingbox.height), Scalar(0, 255, 0), 1, 8, 0);

        for(int i = 0; i < global_params.landmark_num; i++)
        {
             circle(img,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,255,255),-1,8,0);
        }
    }
    cv::imshow( "result", img );
    char a = waitKey(0);
    if (a == 's') {
        save_count++;
        imwrite(to_string(save_count) + ".jpg", img);
    }
}


int FaceDetectionAndAlignment(const char* inputname) {
    extern string cascadeName;
    string inputName;
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    bool tryflip = false;
    double scale  = 1.3;
    CascadeClassifier cascade;

    if (inputname != NULL) {
        inputName.assign(inputname);
    }

    if ( inputName.size() ) {
        if (inputName.find(".jpg") != string::npos || inputName.find(".png") != string::npos
                || inputName.find(".bmp") != string::npos) {
            image = imread( inputName, 1 );
            if (image.empty()) {
                cout << "Read Image fail" << endl;
                return -1;
            }
            else
            {
                std::cout << "Read Image success" << std::endl;
            }

        }
    }

    // -- 0. Load LBF model
    LBFRegressor regressor;
    regressor.Load(modelPath + "LBF.model");

    // -- 1. Load the cascades
    if ( !cascade.load( cascadeName ) ) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if ( !image.empty() ) {
        cout << "In image read" << endl;
        detectAndDraw( image, cascade, regressor,  scale, tryflip );
        waitKey(0);
    }
    return 0;
}


