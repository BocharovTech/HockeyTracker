#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("C:/Users/Yan/Desktop/Detect/v1.mp4");
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, binary;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. Преобразование в серый (убираем цвет)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 2. Бинаризация (превращаем в чисто черный и белый)
        // Порог 100 — это пример. Шайба темная, коврик светлый.
        // Если значение пикселя меньше 100, он станет черным (0), иначе белым (255).
        cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY_INV);

        // Показываем результат (полезно для отладки)
        cv::imshow("Binary view", binary);

        if (cv::waitKey(30) == 'q') break;
    }
    return 0;
}