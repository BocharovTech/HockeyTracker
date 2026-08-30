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
    setWindowTitle("Трекер шайбы");
    setMinimumSize(900, 650);
    resize(1100, 700);

    ui->centralwidget->setStyleSheet(
        "QWidget { background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #0d1320, stop:1 #111827); color: #edf3ff; }");

    QVBoxLayout *rootLayout = new QVBoxLayout(ui->centralwidget);
    rootLayout->setContentsMargins(12, 12, 12, 12);
    rootLayout->setSpacing(10);

    QHBoxLayout *toolbar = new QHBoxLayout();
    toolbar->setContentsMargins(0, 0, 0, 0);
    toolbar->setSpacing(10);

    loadVideoButton = new QPushButton("Загрузить видео", this);
    playPauseButton = new QPushButton("Пауза", this);
    statusLabel = new QLabel("Готово", this);

    loadVideoButton->setStyleSheet(
        "QPushButton { background: #1d4ed8; color: white; border: none; border-radius: 10px; padding: 10px 18px; font-weight: 600; }"
        "QPushButton:hover { background: #2563eb; }"
        "QPushButton:pressed { background: #1e40af; }");

    playPauseButton->setStyleSheet(
        "QPushButton { background: #0f766e; color: white; border: none; border-radius: 10px; padding: 10px 18px; font-weight: 600; }"
        "QPushButton:hover { background: #115e59; }"
        "QPushButton:pressed { background: #134e4a; }");

    statusLabel->setStyleSheet(
        "QLabel { color: #eaf1ff; background: rgba(17, 24, 39, 170); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 10px; padding: 8px 14px; font-weight: 600; }");

    toolbar->addWidget(loadVideoButton);
    toolbar->addWidget(playPauseButton);
    toolbar->addStretch();
    toolbar->addWidget(statusLabel);

    ui->videoLabel->setMinimumSize(640, 480);
    ui->videoLabel->setScaledContents(false);
    ui->videoLabel->setAlignment(Qt::AlignCenter);
    ui->videoLabel->setStyleSheet(
        "QLabel { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #050b12, stop:1 #111827); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 14px; color: #dbeafe; }");

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
    statusLabel->setText(QString("Загружено: %1").arg(fileName));
}

void MainWindow::togglePlayback()
{
    paused.store(!paused.load());
    if (playPauseButton)
        playPauseButton->setText(paused.load() ? "Продолжить" : "Пауза");
    if (statusLabel)
        statusLabel->setText(paused.load() ? "Пауза" : "Воспроизведение");
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
        status = "Калибровка...";
    }
    else
    {
        std::ostringstream s;
        s << "Кадры/с: " << std::fixed << std::setprecision(1) << packet.fps
          << " | Шайбы: " << packet.trackedObjects
          << " | Скорость: " << std::fixed << std::setprecision(1) << packet.speed;
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