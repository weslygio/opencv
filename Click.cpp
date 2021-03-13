#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
extern "C" {
#include <xdo.h>
}
#include <chrono>

using namespace cv;
using namespace std;

Mat src, src_hsv, src_thresh, src_ed, src_edroi;

const String window_detection_name = "Camera";
const String window_settings = "Settings";

float ratio_conhull;

const int max_value_H = 180;
const int max_value = 255;
int low_H = 0, low_S = 0, low_V = 0;
int high_H = 25, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_settings, low_H);
}

static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_settings, high_H);
}

static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_settings, low_S);
}

static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_settings, high_S);
}

static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_settings, low_V);
}

static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_settings, high_V);
}

void TrackbarHSV()
{
    createTrackbar("Low H", window_settings, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_settings, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_settings, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_settings, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_settings, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_settings, &high_V, max_value, on_high_V_thresh_trackbar);
}


const int max_value_xroi = 640;
const int max_value_yroi = 480;
int leftroi = 100, rightroi = 500, uproi = 100, downroi = 300;

static void roi_left(int, void*)
{
    leftroi=min(leftroi,rightroi-1);
    setTrackbarPos("Left side of ROI", window_settings, leftroi);
}

static void roi_right(int, void*)
{
    rightroi=max(leftroi+1,rightroi);
    setTrackbarPos("Right side of ROI", window_settings, rightroi);
}

static void roi_up(int, void*)
{
    uproi=min(uproi,downroi-1);
    setTrackbarPos("Top side of ROI", window_settings, uproi);
}

static void roi_down(int, void*)
{
    downroi=max(uproi+1,downroi);
    setTrackbarPos("Bottom side of ROI", window_settings, downroi);
}

void TrackbarROI()
{
    createTrackbar("Left side of ROI", window_settings, &leftroi, max_value_xroi, roi_left);
    createTrackbar("Right side of ROI", window_settings, &rightroi, max_value_xroi, roi_right);
    createTrackbar("Top side of ROI", window_settings, &uproi, max_value_yroi, roi_up);
    createTrackbar("Bottom side of ROI", window_settings, &downroi, max_value_yroi, roi_down);
}


void contour();
void process();


int main( int argc, char** argv )
{
    VideoCapture cap(0);

    xdo_t * x = xdo_new(NULL);

    namedWindow(window_detection_name);
    namedWindow(window_settings);

    while(1)
    {
        cap >> src;
        if(src.empty()) {break;}
        process();

        //ENDKEY
        char key = (char) waitKey(30);
        if (key == 's') 
        {
            break;
        }
        
    }

    sleep(3);

    while(1)
    {
        cap >> src;

        process();
        contour();

        if (ratio_conhull>0.85)
        {
            xdo_click_window(x, CURRENTWINDOW, 1);
            chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();
            int ms=0;
            while (ms<300)
            {   
                char key = (char) waitKey(30);
                if (key == 'q')
                {
                    break;
                }
                cap >> src;
                process();
                chrono::time_point<chrono::system_clock> fin = chrono::system_clock::now();
                auto milliseconds = chrono::duration_cast<chrono::milliseconds>(fin - start);
                ms = milliseconds.count();
            }
        }

        //ENDKEY
        char key = (char) waitKey(30);
        if (key == 'q')
        {
            break;
        }
    }
    return 0;
}


void contour()
{
    vector<vector<Point> > contours;
    findContours( src_edroi, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<vector<Point> > hull ( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        convexHull( contours[i], hull[i] );
    }

    if (contours.size() !=0)
    {
        float maxcon = contourArea(contours[0]);
        float maxhull = contourArea(hull[0]);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            if (maxcon < contourArea(contours[i]) ){maxcon = contourArea(contours[i]);}
            if (maxhull < contourArea(hull[i]) ) {maxhull = contourArea(hull[i]);}
        }

        ratio_conhull = maxcon/maxhull;
    }

    else if (contours.size() == 0)
    {
        ratio_conhull = 0;
    }
}

void process()
{
    //BGR TO HSV
    cvtColor(src, src_hsv,COLOR_BGR2HSV);


    //HSV TO BW THRESHOLD
    TrackbarHSV();
    inRange(src_hsv, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), src_thresh);

    //BW THRESHOLD ERODING + DILATING
    Mat element = getStructuringElement( 2, Size( 10, 10));
    erode( src_thresh, src_ed, element );
    dilate( src_ed , src_ed , element );
    dilate( src_ed, src_ed , element );
    erode( src_ed , src_ed, element );

    //REGION OF INTEREST
    TrackbarROI();
    Rect roi(leftroi, uproi, rightroi-leftroi, downroi-uproi);
    src_edroi = src_ed(roi);
    rectangle(src_ed, Point(leftroi-1, uproi-1), Point(rightroi+1, downroi+1), Scalar(255,255,255));

    //SHOW VIDEO
    imshow(window_detection_name, src_ed);
        
}