#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
 
using namespace cv;
using namespace std;
#define PI 3.14159265

void detectface( Mat frame );
//global variables
String face_cascade_name = "lbpcascade_frontalface.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";

//function to rotate image and points

void rotateimage(cv::Mat& src, double angle, Point p, vector<Point> &result)
{
    
    cv::Mat r = cv::getRotationMatrix2D(p, angle, 1.0);

    cv::warpAffine(src, src, r, src.size());
    double angle_rad = angle * PI/180;
    for(int i=1;i<result.size();i++)
    {
    	Point original_pt = result[i] - result[0];
    	Point temp3 ;

    	temp3.x = original_pt.x * cos(angle_rad) + original_pt.y * sin(angle_rad);
    	temp3.y = -original_pt.x * sin(angle_rad) + original_pt.y * cos(angle_rad);
    	result[i] = temp3 + result[0];
    }
}
 
int main()
{
 
 //parameters

	int blockSize = 20;
  int apertureSize = 3;
  double k = 0.04;

    
    Mat input,inputgray;
    input = imread( "/home/yatin/Documents/assignment/Images/Image3.jpg");
    cvtColor( input, inputgray, CV_BGR2GRAY );
    int thresh = 220;
    Mat dst, dst_norm, dst_norm_scaled;
   dst = Mat::zeros( input.size(), CV_32FC1 );
   Mat inputbinary(inputgray.size(), inputgray.type());
   // converting image to binary
   threshold(inputgray, inputbinary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    namedWindow( "Display window1", CV_WINDOW_NORMAL);
  imshow("Display window1", inputbinary );
  // eroding to fill the gaps
    Mat erosion_dst;
    int morph_size = 3;
    // Create a structuring element (SE)
    Mat element = getStructuringElement( MORPH_RECT, Size( 4*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    erode( inputbinary, erosion_dst, element );
    bitwise_not ( erosion_dst, erosion_dst );
    
// display the eroded picture
    namedWindow( "Display window", CV_WINDOW_NORMAL);
    imshow("Display window", erosion_dst );

 // finding contours in the eroded image
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    std::vector<std::vector<cv::Point> > contours;
    findContours( erosion_dst, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
 // finding contour with largest area
    double largest_area = 0;
    int largest_contour_index=0;
    Rect bounding_rect;
    for( int i = 0; i< contours.size(); i++)
    {
        //  Find the area of contour
        double a=contourArea( contours[i],false); 
        if(a>largest_area){
            largest_area=a; //cout<<i<<" area  "<<a<<endl;
            // Store the index of largest contour
            largest_contour_index=i;               
            // Find the bounding rectangle for biggest contour
            bounding_rect=boundingRect(contours[i]);
        }
    }

// fitting the largest contour to an approximate polygon i.e. rectangle in our case

    vector<Point> approxCurve;    // stores the minimum points to approximate contour, threhold is adjusted to get 4 corner points
    approxPolyDP(contours[largest_contour_index], approxCurve, 30, true);
   

    //calculating length(greater side) of rectangle
    Point temp = approxCurve[0];
    vector<double> dist;
    for(int i=1;i<approxCurve.size();i++)
    {
    	dist.push_back(  sqrt ( pow( (temp.x - approxCurve[i].x),2) + pow( (temp.y - approxCurve[i].y),2) ) );
    }
    double temp1= *max_element(dist.begin(),dist.end());
    double maxi =0;
    int index_long;
    for(int i=0;i<dist.size();i++)
    { 
    	if(maxi<dist[i] && dist[i]!=temp1)
    	{
    		maxi = dist[i];
    		index_long = i+1;

    	}    
	}
	Point center;
    center = Point(approxCurve[index_long].x,approxCurve[index_long].y);
    double y = temp.y - approxCurve[index_long].y;
    double x = approxCurve[index_long].x - temp.x;
    double theta = atan2(y,x) * 180 / PI;   // calculating angle of rotation
    vector<Point> result ;
    result = approxCurve;
    // rotating the image according to different cases
    if (theta > 0)
    {
    	if(theta > 90)
    	{
    		rotateimage(input, theta, temp,result);
    	}
    	else
    	{
    		rotateimage(input,-1 * theta, temp,result);
    	}
    }
    else
    {
    	if(abs(theta) > 90)
    	{
    		rotateimage(input, abs(theta)-180, temp,result);
    	}
    	else
    	{
    		rotateimage(input, abs(theta),temp,result);
    	}
    }


	for(int i=0;i<result.size();i++)
	{
    center  = Point(result[i].x,result[i].y);
    circle(input, center,10,CV_RGB(255,0,0),3);
    }
    Rect approx = boundingRect(result);
    rectangle(input, approx,  Scalar(0,255,0),2, 8,0);

    Mat cropped_image = input(approx).clone();
    imwrite( "/home/yatin/Documents/assignment/Images/cropped_Image3.jpg", cropped_image );
    detectface(cropped_image);

    waitKey(0);
    return 0;
}

void detectface(Mat frame)
{
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return ; };
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.2, 6, 0, Size(30, 30) );
    cout<<"no. of faces"<<faces.size()<<endl;
    for(int i=0;i<faces.size();i++)
    {
        Mat faceROI = frame(faces[i]);
        imwrite("/home/yatin/Documents/assignment/Images/face_Image3.jpg",faceROI);

        imshow( window_name, faceROI );
        window_name.push_back('i');
    }
}