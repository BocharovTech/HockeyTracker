#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QCloseEvent>
#include <QImage>
#include <QPixmap>

Consumer::Consumer(BlockingQueue<ResultPacket> &queue, std::atomic<bool> &stop)
    : queue_(queue), stop_(stop) {}

void Consumer::run()
{
    ResultPacket packet;
    while (!stop_.load())
    {
        if (!queue_.pop(packet))
        {
            break;
        }
        emit resultReady(packet);
    }
}

// ----- MainWindow -----
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), frames(12), results(12), stop(false)
{
    ui->setupUi(this);
    setWindowTitle("Hockey Tracker");

    std::string videoPath = "../video/v1.mp4";

    reader = new VideoReader(videoPath, frames, stop, 60.0);
    tracker = new TrackerWorker(frames, results, stop);

    reader->start();
    tracker->start();

    consumer = new Consumer(results, stop);
    consumerThread = new QThread(this);
    consumer->moveToThread(consumerThread);

    connect(consumerThread, &QThread::finished, consumer, &QObject::deleteLater);
    connect(consumer, &Consumer::resultReady, this, &MainWindow::onResultReceived);

    QMetaObject::invokeMethod(consumer, &Consumer::run, Qt::QueuedConnection);
    consumerThread->start();
}

MainWindow::~MainWindow()
{
    stop.store(true);
    frames.close();
    results.close();

    if (reader)
    {
        reader->quit();
        reader->wait();
        delete reader;
    }
    if (tracker)
    {
        tracker->quit();
        tracker->wait();
        delete tracker;
    }
    if (consumerThread)
    {
        consumerThread->quit();
        consumerThread->wait();
        delete consumerThread;
    }
    delete ui;
}

void MainWindow::onResultReceived(const ResultPacket &packet)
{
    if (packet.frame.empty())
        return;

    cv::Mat frame = packet.frame;
    if (frame.channels() == 3)
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    }
    QImage img(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(img);

    if (!ui->videoLabel)
    {
        ui->videoLabel = new QLabel(ui->centralwidget);
        ui->videoLabel->setGeometry(0, 0, 800, 600);
        ui->videoLabel->setScaledContents(false);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }
    ui->videoLabel->setPixmap(pixmap.scaled(ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    stop.store(true);
    frames.close();
    results.close();
    event->accept();
}