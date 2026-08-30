#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QThread>
#include <atomic>
#include <string>
#include "tracker.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class Consumer : public QObject
{
    Q_OBJECT
public:
    Consumer(BlockingQueue<ResultPacket> &queue, std::atomic<bool> &stop);
public slots:
    void run();
signals:
    void resultReady(const ResultPacket &packet);

private:
    BlockingQueue<ResultPacket> &queue_;
    std::atomic<bool> &stop_;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void onResultReceived(const ResultPacket &packet);
    void loadVideo();
    void togglePlayback();

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void startWorkers(const std::string &videoPath, int targetFps = 60);
    void stopWorkers();

    Ui::MainWindow *ui;

    BlockingQueue<FramePacket> frames;
    BlockingQueue<ResultPacket> results;
    std::atomic<bool> stop;
    std::atomic<bool> paused;

    VideoReader *reader = nullptr;
    TrackerWorker *tracker = nullptr;

    Consumer *consumer = nullptr;
    QThread *consumerThread = nullptr;

    QPushButton *loadVideoButton = nullptr;
    QPushButton *playPauseButton = nullptr;
    QLabel *statusLabel = nullptr;
};

#endif // MAINWINDOW_H