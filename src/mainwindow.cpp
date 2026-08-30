#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QCloseEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QVBoxLayout>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

static std::string resolveConfigPath()
{
    const std::array<std::string, 3> candidates = {
        "config.ini",
        "../config.ini",
        "src/config.ini"};

    for (const auto &path : candidates)
    {
        std::ifstream config(path);
        if (config.is_open())
            return path;
    }

    return "config.ini";
}

static std::string defaultVideoPath()
{
    const std::string configPath = resolveConfigPath();
    std::ifstream config(configPath);
    if (config.is_open())
    {
        std::string line;
        while (std::getline(config, line))
        {
            if (line.rfind("video_path=", 0) == 0)
                return line.substr(11);
        }
    }
    return "../video/v1.mp4";
}

static int defaultTargetFps()
{
    const std::string configPath = resolveConfigPath();
    std::ifstream config(configPath);
    if (config.is_open())
    {
        std::string line;
        while (std::getline(config, line))
        {
            if (line.rfind("target_fps=", 0) == 0)
            {
                try
                {
                    return std::stoi(line.substr(11));
                }
                catch (...)
                {
                }
            }
        }
    }
    return 30;
}

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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), frames(12), results(12), stop(false), paused(false)
{
    ui->setupUi(this);
    setWindowTitle("Hockey Tracker");

    QVBoxLayout *rootLayout = new QVBoxLayout(ui->centralwidget);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(6);

    QHBoxLayout *toolbar = new QHBoxLayout();
    loadVideoButton = new QPushButton("Load video", this);
    playPauseButton = new QPushButton("Pause", this);
    statusLabel = new QLabel("Ready", this);
    statusLabel->setStyleSheet("QLabel { color: #eaf1ff; background: rgba(30,30,30,140); padding: 4px 8px; border-radius: 4px; }");

    toolbar->addWidget(loadVideoButton);
    toolbar->addWidget(playPauseButton);
    toolbar->addStretch();
    toolbar->addWidget(statusLabel);

    ui->videoLabel->setMinimumSize(640, 480);
    ui->videoLabel->setScaledContents(false);
    ui->videoLabel->setAlignment(Qt::AlignCenter);
    ui->videoLabel->setStyleSheet("QLabel { background: #111111; border: 1px solid #333333; border-radius: 6px; }");

    rootLayout->addLayout(toolbar);
    rootLayout->addWidget(ui->videoLabel, 1);

    connect(loadVideoButton, &QPushButton::clicked, this, &MainWindow::loadVideo);
    connect(playPauseButton, &QPushButton::clicked, this, &MainWindow::togglePlayback);

    startWorkers(defaultVideoPath(), defaultTargetFps());
}

MainWindow::~MainWindow()
{
    stopWorkers();
    delete ui;
}

void MainWindow::startWorkers(const std::string &videoPath, int targetFps)
{
    stop.store(false);
    paused.store(false);
    frames.reset();
    results.reset();

    reader = new VideoReader(videoPath, frames, stop, paused, static_cast<double>(targetFps));
    tracker = new TrackerWorker(frames, results, stop, paused);

    consumer = new Consumer(results, stop);
    consumerThread = new QThread(this);
    consumer->moveToThread(consumerThread);

    connect(consumerThread, &QThread::finished, consumer, &QObject::deleteLater);
    connect(consumer, &Consumer::resultReady, this, &MainWindow::onResultReceived);

    QMetaObject::invokeMethod(consumer, &Consumer::run, Qt::QueuedConnection);
    consumerThread->start();

    reader->start();
    tracker->start();
}

void MainWindow::stopWorkers()
{
    stop.store(true);
    frames.close();
    results.close();

    if (reader)
    {
        reader->quit();
        reader->wait();
        delete reader;
        reader = nullptr;
    }
    if (tracker)
    {
        tracker->quit();
        tracker->wait();
        delete tracker;
        tracker = nullptr;
    }
    if (consumerThread)
    {
        consumerThread->quit();
        consumerThread->wait();
        delete consumerThread;
        consumerThread = nullptr;
    }
    consumer = nullptr;
}

void MainWindow::loadVideo()
{
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select video",
        QString::fromStdString(defaultVideoPath()).section('/', 0, -2),
        "Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm)");

    if (fileName.isEmpty())
        return;

    stopWorkers();
    startWorkers(fileName.toStdString(), defaultTargetFps());
    statusLabel->setText(QString("Loaded: %1").arg(fileName));
}

void MainWindow::togglePlayback()
{
    paused.store(!paused.load());
    if (playPauseButton)
        playPauseButton->setText(paused.load() ? "Play" : "Pause");
    if (statusLabel)
        statusLabel->setText(paused.load() ? "Paused" : "Playing");
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
    else if (frame.channels() == 4)
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
    }

    QImage img(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(img);
    ui->videoLabel->setPixmap(pixmap.scaled(ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

    std::string status;
    if (!packet.calibrated)
    {
        status = "Calibration...";
    }
    else
    {
        std::ostringstream s;
        s << "FPS: " << std::fixed << std::setprecision(1) << packet.fps
          << " | Pucks: " << packet.trackedObjects
          << " | Speed: " << std::fixed << std::setprecision(1) << packet.speed;
        status = s.str();
    }
    if (statusLabel)
        statusLabel->setText(QString::fromStdString(status));
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    stopWorkers();
    event->accept();
}