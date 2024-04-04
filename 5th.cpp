#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;


Mat sPrewitt(Mat input_img)
{
    Mat output_img;
    float val_x[9] = { -1,0,1,-1,0,1,-1,0,1 };
    Mat ker_x = Mat_<float>(3, 3, val_x);
    float val_y[9] = { -1,-1,-1,0,0,0,1,1,1 };
    Mat ker_y = Mat_<float>(3, 3, val_y);
    output_img = input_img;

    int height_x = ker_x.rows;
    int width_x = ker_x.cols;

    int height_y = ker_y.rows;
    int width_y = ker_y.cols;

    for (int row = 1; row < input_img.rows - 1; row++)
    {
        for (int col = 1; col < input_img.cols - 1; col++)
        {
            float G_X = 0;
            for (int h = 0; h < height_x; h++)
            {
                for (int w = 0; w < width_x; w++)
                {
                   G_X += input_img.at<uchar>(row + h - 1, col + w - 1) * ker_x.at<float>(h, w);
                }
            }

            float G_Y = 0;
            for (int h = 0; h < height_y; h++)
            {
                for (int w = 0; w < width_y; w++)
                {
                    G_Y += input_img.at<uchar>(row + h - 1, col + w - 1) * ker_y.at<float>(h, w);
                }
            }
            
            output_img.at<uchar>(row, col) = saturate_cast<uchar>(abs(G_X) + abs(G_Y));
            
        }
    }
    return output_img;
}




void RGBtoGRAY(Mat& src, Mat& des) {           //彩色转灰度的函数
	des.create(src.rows, src.cols, CV_8UC1);
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			Vec3b& m = src.at<Vec3b>(r, c);
			int gray = (m[2] * 30 + m[1] * 59 + m[0] * 11 + 50) / 100;
			des.at<uchar>(r, c) = gray;
		}
	}
}
void main()
{
	Mat input = imread("2233.jpg");

	Mat gray;

	RGBtoGRAY(input, gray);    //彩色图转为灰度图
	namedWindow("gray",0);
	imshow("gray", gray);//测试是否转化成功
    Mat tmp;
    Mat result;
    GaussianBlur(gray, tmp, Size(15, 15), 1, 9);// 对图像进行高斯滤波
    namedWindow("tmpHist", 0);
    imshow("tmpHist", tmp);
    result = sPrewitt(tmp);
    namedWindow("result", 0);
    imshow("result", result);
	
	waitKey(0);
}