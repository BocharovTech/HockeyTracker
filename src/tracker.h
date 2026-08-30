#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>

#include <QThread>
#include <QObject>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

// ---------- Константы ----------
extern const int CALIBRATION_FRAMES;
extern const int CALIBRATION_MIN_HITS;
extern const double CALIBRATION_MAX_SPREAD;
extern const double CALIBRATION_CLUSTER_RADIUS;

extern const int MIN_PUCK_RADIUS;
extern const int MAX_PUCK_RADIUS;
extern const double MIN_AREA;
extern const double MAX_AREA;
extern const double MIN_CIRCULARITY;
extern const double STATIC_MIN_CIRCULARITY;
extern const double STATIC_MIN_APPEARANCE;
extern const double STATIC_MAX_COLOR_FRACTION;
extern const double STATIC_MIN_DARK_FRACTION;
extern const double EXPECTED_AREA;
extern const double EXPECTED_RADIUS;

extern const double MOTION_THRESHOLD;
extern const int MAX_MISSED_FRAMES;
extern const int HISTORY_SIZE;

extern const double TEMPLATE_ACCEPT_SCORE;
extern const double TEMPLATE_STRONG_SCORE;
extern const int TEMPLATE_SIZE;

extern const double VELOCITY_DAMPING;
extern const double MAX_SPEED;

// ---------- Структуры данных ----------
struct Detection
{
    cv::Point2f center{0, 0};
    double area = 0;
    double circularity = 0;
    double radius = 21.0;
    double solidity = 0;
    double extent = 0;
    double quality = 0;
    double appearance = 0;
    double darkFraction = 0;
    double colorFraction = 0;
    double textureScore = 0;
    double roundness = 0;
    bool contour = false;
    bool hough = false;
};

struct TimedPoint
{
    cv::Point2f point{0, 0};
    int frame = 0;
};

struct StaticPuck
{
    cv::Point2f center{0, 0};
    double radius = 21.0;
    int hits = 0;
    int missing = 0;
};

struct StaticCluster
{
    cv::Point2f center{0, 0};
    std::vector<cv::Point2f> samples;
    int hits = 0;
    double appearanceSum = 0;
    double darkFractionSum = 0;
    double colorFractionSum = 0;
    double circularitySum = 0;
};

struct MovingTrack
{
    cv::Point2f position{0, 0};
    cv::Point2f previousPosition{0, 0};
    cv::Point2f velocity{0, 0};
    cv::Point2f previousVelocity{0, 0};
    cv::Point2f predicted{0, 0};
    double radius = 21.0;
    double area = 1300.0;
    double confidence = 0;
    int missed = 0;
    int lastFrame = -1;
    bool initialized = false;
    std::deque<TimedPoint> history;
    cv::Mat puckTemplate;
};

struct FramePacket
{
    cv::Mat frame;
    int number = 0;
    double fps = 0.0;
};

struct ResultPacket
{
    cv::Mat frame;
    int number = 0;
    double fps = 0.0;
    int calibrationFrames = 0;
    int totalFrames = 0;
    double speed = 0.0;
    int trackedObjects = 0;
    std::vector<StaticPuck> statics;
    MovingTrack moving;
    bool calibrated = false;
};

// ---------- Шаблонный класс блокирующей очереди ----------
template <typename T>
class BlockingQueue
{
public:
    explicit BlockingQueue(size_t maxSize) : maxSize_(maxSize) {}

    bool push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [&]
                      { return closed_ || queue_.size() < maxSize_; });
        if (closed_)
            return false;
        queue_.push_back(std::move(value));
        notEmpty_.notify_one();
        return true;
    }

    bool pop(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [&]
                       { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        value = std::move(queue_.front());
        queue_.pop_front();
        notFull_.notify_one();
        return true;
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        closed_ = false;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::deque<T> queue_;
    size_t maxSize_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
};

// ---------- Поток чтения видео ----------
class VideoReader : public QThread
{
    Q_OBJECT
public:
    VideoReader(const std::string &path,
                BlockingQueue<FramePacket> &output,
                std::atomic<bool> &stop,
                std::atomic<bool> &paused,
                double targetFps = 0.0);

protected:
    void run() override;

private:
    std::string path_;
    BlockingQueue<FramePacket> &output_;
    std::atomic<bool> &stop_;
    std::atomic<bool> &paused_;
    double targetFps_;
};

// ---------- Поток трекера ----------
class TrackerWorker : public QThread
{
    Q_OBJECT
public:
    TrackerWorker(BlockingQueue<FramePacket> &input,
                  BlockingQueue<ResultPacket> &output,
                  std::atomic<bool> &stop,
                  std::atomic<bool> &paused);

protected:
    void run() override;

private:
    BlockingQueue<FramePacket> &input_;
    BlockingQueue<ResultPacket> &output_;
    std::atomic<bool> &stop_;
    std::atomic<bool> &paused_;
};

// ---------- Вспомогательные функции (объявления) ----------
double distanceBetween(const cv::Point2f &a, const cv::Point2f &b);
double magnitude(const cv::Point2f &p);
cv::Point2f clampPoint(const cv::Point2f &p, const cv::Mat &frame);

bool prepareGray(const cv::Mat &frame, cv::Mat &gray);
void makeBinary(const cv::Mat &gray, cv::Mat &binary);
void makeMotionMask(const cv::Mat &previousGray, const cv::Mat &gray, cv::Mat &motion);

double puckAppearanceScore(const cv::Mat &gray, const cv::Mat &hsv,
                           const cv::Point2f &center, double radius,
                           double &darkFraction, double &colorFraction,
                           double &textureScore);

std::vector<Detection> findContours(const cv::Mat &binary, const cv::Mat &gray, const cv::Mat &hsv);
std::vector<Detection> findHough(const cv::Mat &gray, const cv::Mat &hsv);
std::vector<Detection> mergeDetections(const std::vector<Detection> &contours,
                                       const std::vector<Detection> &hough);

double motionFraction(const cv::Mat &motion, const cv::Point2f &center, double radius);

void updateStaticClusters(std::vector<StaticCluster> &clusters,
                          const std::vector<Detection> &detections);
double clusterSpread(const StaticCluster &cluster);
std::vector<StaticPuck> buildStaticPucks(const std::vector<StaticCluster> &clusters,
                                         int frames);

bool nearStatic(const cv::Point2f &p, const std::vector<StaticPuck> &statics, double radius);
double nearestStaticDistance(const cv::Point2f &p, const std::vector<StaticPuck> &statics);
void updateStaticTracks(std::vector<StaticPuck> &statics,
                        const std::vector<Detection> &detections);

bool extractTemplate(const cv::Mat &gray, const cv::Point2f &center,
                     int size, cv::Mat &templ);
void updateTemplate(MovingTrack &track, const cv::Mat &gray, const cv::Point2f &center);
bool localTemplateMatch(const cv::Mat &gray, const cv::Mat &templ,
                        const cv::Point2f &center, double searchRadius,
                        cv::Point2f &bestCenter, double &bestScore);

void updateMoving(MovingTrack &track, const Detection &d, const cv::Mat &gray,
                  int frame, double confidence, bool updateTempl);
void missMoving(MovingTrack &track, const cv::Mat &frame);

double directionScore(const MovingTrack &track, const cv::Point2f &candidate);
int chooseMovingDetection(const std::vector<Detection> &detections,
                          const cv::Mat &motion,
                          const MovingTrack &track,
                          const std::vector<StaticPuck> &statics);

bool recoverByTemplate(const cv::Mat &gray, const cv::Mat &motion,
                       MovingTrack &track, const std::vector<StaticPuck> &statics,
                       int frame);
bool recoverByLocalHough(const cv::Mat &gray, const cv::Mat &motion,
                         MovingTrack &track, const std::vector<StaticPuck> &statics,
                         int frame);

void drawStatic(cv::Mat &frame, const StaticPuck &puck, int id);
void drawMoving(cv::Mat &frame, const MovingTrack &track, int id);

#endif // TRACKER_H