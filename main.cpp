#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture cap("D:/Qt/QtProjects/HockeyTracker/v1.mp4");
    if (!cap.isOpened())
        return -1;

    cv::Mat frame, gray, binary;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY_INV);

        cv::imshow("Binary view", binary);

        if (cv::waitKey(30) == 'q')
            break;
    }
    return 0;
}