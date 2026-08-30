#include "tracker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

// ---------- Константы (определения) ----------
const int CALIBRATION_FRAMES = 100;
const int CALIBRATION_MIN_HITS = 60;
const double CALIBRATION_MAX_SPREAD = 11.0;
const double CALIBRATION_CLUSTER_RADIUS = 34.0;

const int MIN_PUCK_RADIUS = 12;
const int MAX_PUCK_RADIUS = 34;
const double MIN_AREA = 300.0;
const double MAX_AREA = 2600.0;
const double MIN_CIRCULARITY = 0.22;
const double STATIC_MIN_CIRCULARITY = 0.50;
const double STATIC_MIN_APPEARANCE = 0.50;
const double STATIC_MAX_COLOR_FRACTION = 0.35;
const double STATIC_MIN_DARK_FRACTION = 0.20;
const double EXPECTED_AREA = 1300.0;
const double EXPECTED_RADIUS = 21.0;

const double MOTION_THRESHOLD = 18.0;
const int MAX_MISSED_FRAMES = 45;
const int HISTORY_SIZE = 12;

const double TEMPLATE_ACCEPT_SCORE = 0.52;
const double TEMPLATE_STRONG_SCORE = 0.64;
const int TEMPLATE_SIZE = 56;

const double VELOCITY_DAMPING = 0.94;
const double MAX_SPEED = 55.0;

// ---------- Вспомогательные функции ----------
double distanceBetween(const cv::Point2f &a, const cv::Point2f &b)
{
    return std::hypot(static_cast<double>(a.x) - b.x,
                      static_cast<double>(a.y) - b.y);
}

double magnitude(const cv::Point2f &p)
{
    return std::hypot(static_cast<double>(p.x), static_cast<double>(p.y));
}

cv::Point2f clampPoint(const cv::Point2f &p, const cv::Mat &frame)
{
    return cv::Point2f(
        std::clamp(p.x, 0.0f, static_cast<float>(frame.cols - 1)),
        std::clamp(p.y, 0.0f, static_cast<float>(frame.rows - 1)));
}

bool prepareGray(const cv::Mat &frame, cv::Mat &gray)
{
    if (frame.empty())
        return false;
    if (frame.channels() == 1)
        gray = frame.clone();
    else if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    else
        return false;
    return true;
}

void makeBinary(const cv::Mat &gray, cv::Mat &binary)
{
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, binary, 100, 255, cv::THRESH_BINARY_INV);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
}

void makeMotionMask(const cv::Mat &previousGray, const cv::Mat &gray, cv::Mat &motion)
{
    if (previousGray.empty() || previousGray.size() != gray.size())
    {
        motion = cv::Mat::zeros(gray.size(), CV_8UC1);
        return;
    }
    cv::Mat a, b, diff;
    cv::GaussianBlur(previousGray, a, cv::Size(5, 5), 0);
    cv::GaussianBlur(gray, b, cv::Size(5, 5), 0);
    cv::absdiff(a, b, diff);
    cv::threshold(diff, motion, MOTION_THRESHOLD, 255, cv::THRESH_BINARY);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(motion, motion, kernel);
}

double puckAppearanceScore(const cv::Mat &gray, const cv::Mat &hsv,
                           const cv::Point2f &center, double radius,
                           double &darkFraction, double &colorFraction,
                           double &textureScore)
{
    darkFraction = 0.0;
    colorFraction = 0.0;
    textureScore = 0.0;
    if (gray.empty() || hsv.empty())
        return 0.0;

    const int r = std::max(8, static_cast<int>(std::round(radius)));
    const int outer = std::max(r + 2, static_cast<int>(std::round(radius * 1.45)));
    const int cx = static_cast<int>(std::round(center.x));
    const int cy = static_cast<int>(std::round(center.y));

    const int x0 = std::max(0, cx - outer);
    const int y0 = std::max(0, cy - outer);
    const int x1 = std::min(gray.cols - 1, cx + outer);
    const int y1 = std::min(gray.rows - 1, cy + outer);
    if (x1 <= x0 || y1 <= y0)
        return 0.0;

    const cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    cv::Mat grayRoi = gray(roi);
    cv::Mat hsvRoi = hsv(roi);

    cv::Mat innerMask(grayRoi.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat ringMask(grayRoi.size(), CV_8UC1, cv::Scalar(0));
    cv::circle(innerMask,
               cv::Point(cx - x0, cy - y0),
               std::max(5, static_cast<int>(std::round(radius * 0.72))),
               cv::Scalar(255), -1);
    cv::circle(ringMask,
               cv::Point(cx - x0, cy - y0),
               std::max(7, static_cast<int>(std::round(radius * 1.40))),
               cv::Scalar(255), -1);
    cv::circle(ringMask,
               cv::Point(cx - x0, cy - y0),
               std::max(5, static_cast<int>(std::round(radius * 1.08))),
               cv::Scalar(0), -1);

    const int innerPixels = cv::countNonZero(innerMask);
    const int ringPixels = cv::countNonZero(ringMask);
    if (innerPixels < 20 || ringPixels < 20)
        return 0.0;

    const double innerMean = cv::mean(grayRoi, innerMask)[0];
    const double ringMean = cv::mean(grayRoi, ringMask)[0];
    cv::Scalar meanValue, stdValue;
    cv::meanStdDev(grayRoi, meanValue, stdValue, innerMask);

    cv::Mat darkMask;
    cv::threshold(grayRoi, darkMask, 105, 255, cv::THRESH_BINARY_INV);
    darkFraction = static_cast<double>(cv::countNonZero(darkMask & innerMask)) /
                   static_cast<double>(innerPixels);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvRoi, hsvChannels);
    cv::Mat coloredMask;
    cv::inRange(hsvRoi, cv::Scalar(0, 75, 0), cv::Scalar(180, 255, 255), coloredMask);
    colorFraction = static_cast<double>(cv::countNonZero(coloredMask & innerMask)) /
                    static_cast<double>(innerPixels);

    const double contrast = std::clamp((ringMean - innerMean - 18.0) / 85.0, 0.0, 1.0);
    const double darkness = std::clamp((darkFraction - 0.35) / 0.55, 0.0, 1.0);
    const double lowColor = 1.0 - std::clamp((colorFraction - 0.05) / 0.45, 0.0, 1.0);
    textureScore = 1.0 - std::clamp((stdValue[0] - 10.0) / 48.0, 0.0, 1.0);

    return std::clamp(0.42 * contrast +
                          0.30 * darkness +
                          0.18 * lowColor +
                          0.10 * textureScore,
                      0.0, 1.0);
}

bool goodStaticCandidate(const Detection &d)
{
    return d.contour &&
           d.roundness >= 0.68 &&
           d.circularity >= STATIC_MIN_CIRCULARITY &&
           d.appearance >= STATIC_MIN_APPEARANCE &&
           d.darkFraction >= STATIC_MIN_DARK_FRACTION &&
           d.colorFraction <= STATIC_MAX_COLOR_FRACTION &&
           d.solidity >= 0.72 &&
           d.extent >= 0.48;
}

std::vector<Detection> findContours(const cv::Mat &binary, const cv::Mat &gray, const cv::Mat &hsv)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Detection> result;
    for (const auto &contour : contours)
    {
        const double area = cv::contourArea(contour);
        if (area < MIN_AREA || area > MAX_AREA)
            continue;

        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0)
            continue;

        const cv::Rect rect = cv::boundingRect(contour);
        if (rect.width <= 0 || rect.height <= 0)
            continue;

        const double aspect = static_cast<double>(rect.width) / rect.height;
        if (aspect < 0.45 || aspect > 2.2)
            continue;

        const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity < MIN_CIRCULARITY)
            continue;

        const cv::Moments m = cv::moments(contour);
        if (m.m00 == 0)
            continue;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        const double hullArea = cv::contourArea(hull);
        const double solidity = hullArea > 0 ? area / hullArea : 0;
        const double extent = area / static_cast<double>(rect.width * rect.height);
        const double radius = std::sqrt(area / CV_PI);
        cv::Point2f enclosingCenter;
        float enclosingRadius = 0.0f;
        cv::minEnclosingCircle(contour, enclosingCenter, enclosingRadius);
        const double roundness = (enclosingRadius > 0.0f)
                                     ? area / (CV_PI * enclosingRadius * enclosingRadius)
                                     : 0.0;

        if (radius < MIN_PUCK_RADIUS || radius > MAX_PUCK_RADIUS)
            continue;

        Detection d;
        d.center = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                               static_cast<float>(m.m01 / m.m00));
        d.area = area;
        d.circularity = circularity;
        d.radius = radius;
        d.solidity = solidity;
        d.extent = extent;
        d.roundness = roundness;

        const double areaScore = 1.0 - std::min(std::abs(area - EXPECTED_AREA) / EXPECTED_AREA, 1.0);
        const double circularityScore = std::clamp(circularity / 0.90, 0.0, 1.0);
        const double solidityScore = std::clamp(solidity, 0.0, 1.0);
        const double extentScore = std::clamp(extent / 0.78, 0.0, 1.0);

        d.quality = 0.45 * circularityScore +
                    0.25 * areaScore +
                    0.20 * solidityScore +
                    0.10 * extentScore;
        d.appearance = puckAppearanceScore(gray, hsv, d.center, d.radius,
                                           d.darkFraction, d.colorFraction,
                                           d.textureScore);
        d.contour = true;
        result.push_back(d);
    }
    return result;
}

std::vector<Detection> findHough(const cv::Mat &gray, const cv::Mat &hsv)
{
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
                     1.2, 27.0, 80.0, 18.0,
                     MIN_PUCK_RADIUS, MAX_PUCK_RADIUS);

    std::vector<Detection> result;
    for (const auto &c : circles)
    {
        Detection d;
        d.center = cv::Point2f(c[0], c[1]);
        d.radius = c[2];
        d.area = CV_PI * d.radius * d.radius;
        d.quality = 0.55;
        d.appearance = puckAppearanceScore(gray, hsv, d.center, d.radius,
                                           d.darkFraction, d.colorFraction,
                                           d.textureScore);
        d.hough = true;
        result.push_back(d);
    }
    return result;
}

std::vector<Detection> mergeDetections(const std::vector<Detection> &contours,
                                       const std::vector<Detection> &hough)
{
    std::vector<Detection> result = contours;

    for (const auto &h : hough)
    {
        int best = -1;
        double bestDist = std::numeric_limits<double>::max();

        for (size_t i = 0; i < result.size(); ++i)
        {
            const double d = distanceBetween(h.center, result[i].center);
            if (d < 22.0 && d < bestDist)
            {
                bestDist = d;
                best = static_cast<int>(i);
            }
        }

        if (best >= 0)
        {
            result[best].hough = true;
            result[best].radius = 0.7 * result[best].radius + 0.3 * h.radius;
            result[best].quality = std::max(result[best].quality,
                                            0.65 * h.quality + 0.35 * result[best].quality);
            result[best].appearance = std::max(result[best].appearance, h.appearance);
            result[best].darkFraction = std::max(result[best].darkFraction, h.darkFraction);
            result[best].colorFraction = std::min(result[best].colorFraction, h.colorFraction);
            result[best].textureScore = std::max(result[best].textureScore, h.textureScore);
            result[best].roundness = std::max(result[best].roundness, h.roundness);
        }
        else
        {
            result.push_back(h);
        }
    }

    std::vector<Detection> filtered;
    for (const auto &d : result)
    {
        bool duplicate = false;
        for (const auto &existing : filtered)
        {
            if (distanceBetween(d.center, existing.center) < 20.0)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            filtered.push_back(d);
    }
    return filtered;
}

double motionFraction(const cv::Mat &motion, const cv::Point2f &center, double radius)
{
    if (motion.empty())
        return 0;

    const int r = std::max(8, static_cast<int>(radius * 1.35));
    const int x0 = std::max(0, static_cast<int>(center.x) - r);
    const int y0 = std::max(0, static_cast<int>(center.y) - r);
    const int x1 = std::min(motion.cols - 1, static_cast<int>(center.x) + r);
    const int y1 = std::min(motion.rows - 1, static_cast<int>(center.y) + r);

    if (x1 <= x0 || y1 <= y0)
        return 0;

    const cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    return static_cast<double>(cv::countNonZero(motion(roi))) /
           static_cast<double>(roi.area());
}

void updateStaticClusters(std::vector<StaticCluster> &clusters,
                          const std::vector<Detection> &detections)
{
    std::vector<bool> used(clusters.size(), false);

    for (const auto &d : detections)
    {
        if (d.quality < 0.45 || !goodStaticCandidate(d))
            continue;

        int best = -1;
        double bestDist = CALIBRATION_CLUSTER_RADIUS;

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            const double dist = distanceBetween(d.center, clusters[i].center);
            if (!used[i] && dist < bestDist)
            {
                bestDist = dist;
                best = static_cast<int>(i);
            }
        }

        if (best < 0)
        {
            StaticCluster c;
            c.center = d.center;
            c.samples.push_back(d.center);
            c.hits = 1;
            c.appearanceSum = d.appearance;
            c.darkFractionSum = d.darkFraction;
            c.colorFractionSum = d.colorFraction;
            c.circularitySum = d.circularity;
            clusters.push_back(std::move(c));
            used.push_back(true);
        }
        else
        {
            StaticCluster &c = clusters[best];
            c.samples.push_back(d.center);
            c.hits++;
            c.appearanceSum += d.appearance;
            c.darkFractionSum += d.darkFraction;
            c.colorFractionSum += d.colorFraction;
            c.circularitySum += d.circularity;
            c.center.x = static_cast<float>(c.center.x * 0.85 + d.center.x * 0.15);
            c.center.y = static_cast<float>(c.center.y * 0.85 + d.center.y * 0.15);
            used[best] = true;

            if (c.samples.size() > 250)
                c.samples.erase(c.samples.begin(), c.samples.begin() + 50);
        }
    }
}

double clusterSpread(const StaticCluster &cluster)
{
    if (cluster.samples.empty())
        return std::numeric_limits<double>::max();
    double sum = 0;
    for (const auto &p : cluster.samples)
        sum += distanceBetween(p, cluster.center);
    return sum / cluster.samples.size();
}

std::vector<StaticPuck> buildStaticPucks(const std::vector<StaticCluster> &clusters,
                                         int frames)
{
    std::vector<StaticPuck> result;

    for (const auto &c : clusters)
    {
        if (c.hits < CALIBRATION_MIN_HITS)
            continue;

        const double coverage = static_cast<double>(c.hits) / std::max(1, frames);
        const double spread = clusterSpread(c);
        const double avgAppearance = c.appearanceSum / std::max(1, c.hits);
        const double avgDark = c.darkFractionSum / std::max(1, c.hits);
        const double avgColor = c.colorFractionSum / std::max(1, c.hits);
        const double avgCircularity = c.circularitySum / std::max(1, c.hits);

        if (coverage < 0.30 || spread > CALIBRATION_MAX_SPREAD)
            continue;
        if (avgAppearance < STATIC_MIN_APPEARANCE ||
            avgDark < STATIC_MIN_DARK_FRACTION ||
            avgColor > STATIC_MAX_COLOR_FRACTION ||
            avgCircularity < STATIC_MIN_CIRCULARITY)
            continue;

        StaticPuck puck;
        puck.center = c.center;
        puck.radius = EXPECTED_RADIUS;
        puck.hits = c.hits;
        result.push_back(puck);
    }

    std::sort(result.begin(), result.end(),
              [](const StaticPuck &a, const StaticPuck &b)
              { return a.hits > b.hits; });

    std::vector<StaticPuck> unique;
    for (const auto &p : result)
    {
        bool duplicate = false;
        for (const auto &u : unique)
        {
            if (distanceBetween(p.center, u.center) < 35.0)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            unique.push_back(p);
    }
    return unique;
}

bool nearStatic(const cv::Point2f &p, const std::vector<StaticPuck> &statics, double radius)
{
    for (const auto &s : statics)
        if (distanceBetween(p, s.center) < radius)
            return true;
    return false;
}

double nearestStaticDistance(const cv::Point2f &p, const std::vector<StaticPuck> &statics)
{
    if (statics.empty())
        return 9999.0;
    double best = 9999.0;
    for (const auto &s : statics)
        best = std::min(best, distanceBetween(p, s.center));
    return best;
}

void updateStaticTracks(std::vector<StaticPuck> &statics,
                        const std::vector<Detection> &detections)
{
    for (auto &s : statics)
    {
        int best = -1;
        double bestDist = 30.0;
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const double d = distanceBetween(s.center, detections[i].center);
            if (d < bestDist)
            {
                bestDist = d;
                best = static_cast<int>(i);
            }
        }

        if (best >= 0)
        {
            s.center.x = static_cast<float>(s.center.x * 0.9 + detections[best].center.x * 0.1);
            s.center.y = static_cast<float>(s.center.y * 0.9 + detections[best].center.y * 0.1);
            s.radius = 0.9 * s.radius + 0.1 * detections[best].radius;
            s.hits++;
            s.missing = 0;
        }
        else
        {
            s.missing++;
        }
    }
}

bool extractTemplate(const cv::Mat &gray, const cv::Point2f &center,
                     int size, cv::Mat &templ)
{
    if (gray.empty())
        return false;
    const int half = size / 2;
    const int cx = static_cast<int>(std::round(center.x));
    const int cy = static_cast<int>(std::round(center.y));
    const int x = cx - half, y = cy - half;
    if (x < 0 || y < 0 || x + size > gray.cols || y + size > gray.rows)
        return false;
    templ = gray(cv::Rect(x, y, size, size)).clone();
    return true;
}

void updateTemplate(MovingTrack &track, const cv::Mat &gray, const cv::Point2f &center)
{
    cv::Mat current;
    if (!extractTemplate(gray, center, TEMPLATE_SIZE, current))
        return;
    if (track.puckTemplate.empty())
        track.puckTemplate = current;
    else
        cv::addWeighted(track.puckTemplate, 0.90, current, 0.10, 0.0, track.puckTemplate);
}

bool localTemplateMatch(const cv::Mat &gray, const cv::Mat &templ,
                        const cv::Point2f &center, double searchRadius,
                        cv::Point2f &bestCenter, double &bestScore)
{
    if (templ.empty() || gray.empty())
        return false;

    const int halfTemplate = templ.cols / 2;
    const int r = static_cast<int>(searchRadius) + halfTemplate;
    const int cx = static_cast<int>(center.x), cy = static_cast<int>(center.y);

    const int x = std::max(0, cx - r);
    const int y = std::max(0, cy - r);
    const int x2 = std::min(gray.cols, cx + r);
    const int y2 = std::min(gray.rows, cy + r);

    if (x2 - x < templ.cols || y2 - y < templ.rows)
        return false;

    cv::Mat roi = gray(cv::Rect(x, y, x2 - x, y2 - y));
    cv::Mat result;
    cv::matchTemplate(roi, templ, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    bestScore = maxVal;
    bestCenter = cv::Point2f(static_cast<float>(x + maxLoc.x + templ.cols / 2),
                             static_cast<float>(y + maxLoc.y + templ.rows / 2));
    return true;
}

void updateMoving(MovingTrack &track, const Detection &d, const cv::Mat &gray,
                  int frame, double confidence, bool updateTempl)
{
    if (!track.initialized)
    {
        track.position = d.center;
        track.previousPosition = d.center;
        track.predicted = d.center;
        track.velocity = cv::Point2f(0, 0);
        track.radius = d.radius;
        track.area = d.area;
        track.confidence = confidence;
        track.missed = 0;
        track.lastFrame = frame;
        track.initialized = true;
        track.history.push_back({d.center, frame});
        if (updateTempl)
            updateTemplate(track, gray, d.center);
        return;
    }

    const int dtFrames = std::max(1, frame - track.lastFrame);
    const double dt = static_cast<double>(dtFrames);

    const cv::Point2f delta = d.center - track.position;
    cv::Point2f measuredVelocity(static_cast<float>(delta.x / dt),
                                 static_cast<float>(delta.y / dt));

    if (magnitude(measuredVelocity) > MAX_SPEED)
    {
        const double scale = MAX_SPEED / magnitude(measuredVelocity);
        measuredVelocity.x = static_cast<float>(measuredVelocity.x * scale);
        measuredVelocity.y = static_cast<float>(measuredVelocity.y * scale);
    }

    track.previousPosition = track.position;
    track.previousVelocity = track.velocity;
    track.position = d.center;
    track.velocity.x = static_cast<float>(track.velocity.x * 0.35 + measuredVelocity.x * 0.65);
    track.velocity.y = static_cast<float>(track.velocity.y * 0.35 + measuredVelocity.y * 0.65);
    track.predicted = track.position + track.velocity;
    track.radius = 0.85 * track.radius + 0.15 * d.radius;
    track.area = 0.85 * track.area + 0.15 * d.area;
    track.missed = 0;
    track.lastFrame = frame;
    track.confidence = std::clamp(0.75 * track.confidence + 0.25 * confidence, 0.0, 1.0);

    track.history.push_back({d.center, frame});
    while (static_cast<int>(track.history.size()) > HISTORY_SIZE)
        track.history.pop_front();

    if (updateTempl)
        updateTemplate(track, gray, d.center);
}

void missMoving(MovingTrack &track, const cv::Mat &frame)
{
    if (!track.initialized)
        return;
    track.missed++;
    track.velocity.x = static_cast<float>(track.velocity.x * VELOCITY_DAMPING);
    track.velocity.y = static_cast<float>(track.velocity.y * VELOCITY_DAMPING);
    track.position = clampPoint(track.position + track.velocity, frame);
    track.predicted = clampPoint(track.position + track.velocity, frame);
    track.confidence *= 0.92;
}

double directionScore(const MovingTrack &track, const cv::Point2f &candidate)
{
    if (track.history.size() < 3 || magnitude(track.velocity) < 1.0)
        return 0.5;
    const cv::Point2f toCandidate = candidate - track.position;
    const double len = magnitude(toCandidate);
    const double speed = magnitude(track.velocity);
    if (len < 2.0 || speed < 1.0)
        return 0.5;
    const double dot = (toCandidate.x * track.velocity.x + toCandidate.y * track.velocity.y) / (len * speed);
    return std::clamp((dot + 1.0) * 0.5, 0.0, 1.0);
}

int chooseMovingDetection(const std::vector<Detection> &detections,
                          const cv::Mat &motion,
                          const MovingTrack &track,
                          const std::vector<StaticPuck> &statics)
{
    if (detections.empty())
        return -1;

    int best = -1;
    double bestScore = -1e9;
    const double currentSpeed = magnitude(track.velocity);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        const Detection &d = detections[i];
        const double motionValue = motionFraction(motion, d.center, d.radius);
        const double predictedDistance = distanceBetween(d.center, track.predicted);
        const double lastDistance = distanceBetween(d.center, track.position);
        const double dir = directionScore(track, d.center);
        const double measuredSpeed = lastDistance;
        const double speedDiff = std::abs(measuredSpeed - currentSpeed);
        const double staticDistance = nearestStaticDistance(d.center, statics);

        if (staticDistance < 24.0)
        {
            const bool strongMotion = motionValue >= 0.12;
            const bool goodPrediction = predictedDistance <= 55.0;
            if (!(strongMotion && goodPrediction))
                continue;
        }

        double score = 0.0;
        score += 2.4 * d.quality;
        score += 3.2 * std::exp(-predictedDistance / 65.0);
        score += 1.0 * std::exp(-lastDistance / 90.0);
        score += 1.8 * dir;
        score += 1.2 * std::exp(-speedDiff / 35.0);
        score += 2.4 * std::clamp(motionValue / 0.20, 0.0, 1.0);

        if (staticDistance < 45.0)
            score -= 1.8 * (45.0 - staticDistance) / 45.0;

        if (track.missed > 0 && predictedDistance > 150.0 && motionValue < 0.06)
            score -= 1.8;

        if (score > bestScore)
        {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }

    if (best < 0)
        return -1;

    const Detection &d = detections[best];
    const double predictedDistance = distanceBetween(d.center, track.predicted);
    const double motionValue = motionFraction(motion, d.center, d.radius);
    const double staticDistance = nearestStaticDistance(d.center, statics);

    if (staticDistance < 24.0 && motionValue < 0.12)
        return -1;

    if (track.missed == 0)
    {
        if (predictedDistance > 150.0 && motionValue < 0.08)
            return -1;
    }
    else if (track.missed > 5)
    {
        if (predictedDistance > 210.0 && motionValue < 0.10 && staticDistance > 55.0)
            return -1;
    }

    return best;
}

bool recoverByTemplate(const cv::Mat &gray, const cv::Mat &motion,
                       MovingTrack &track, const std::vector<StaticPuck> &statics,
                       int frame)
{
    if (track.puckTemplate.empty())
        return false;

    const double searchRadius = std::clamp(
        65.0 + magnitude(track.velocity) * 2.5 + track.missed * 12.0,
        65.0, 260.0);

    cv::Point2f center;
    double score = 0;
    if (!localTemplateMatch(gray, track.puckTemplate, track.predicted, searchRadius,
                            center, score))
        return false;

    const double motionValue = motionFraction(motion, center, track.radius);
    const double staticDistance = nearestStaticDistance(center, statics);

    bool accept = false;
    if (score >= TEMPLATE_STRONG_SCORE)
        accept = true;
    else if (score >= TEMPLATE_ACCEPT_SCORE && motionValue >= 0.08)
        accept = true;
    else if (score >= 0.60 && distanceBetween(center, track.predicted) < 100.0)
        accept = true;

    if (staticDistance < 28.0)
        accept = false;

    if (!accept)
        return false;

    Detection d;
    d.center = center;
    d.radius = track.radius;
    d.area = track.area;
    d.quality = std::clamp(0.45 + score * 0.45 + motionValue * 0.15, 0.0, 1.0);

    updateMoving(track, d, gray, frame,
                 std::clamp(0.45 + score * 0.5, 0.0, 1.0),
                 false);
    return true;
}

bool recoverByLocalHough(const cv::Mat &gray, const cv::Mat &motion,
                         MovingTrack &track, const std::vector<StaticPuck> &statics,
                         int frame)
{
    const int r = static_cast<int>(std::clamp(
        75.0 + magnitude(track.velocity) * 2.0 + track.missed * 10.0,
        75.0, 240.0));

    const int cx = static_cast<int>(track.predicted.x);
    const int cy = static_cast<int>(track.predicted.y);
    const int x = std::max(0, cx - r);
    const int y = std::max(0, cy - r);
    const int x2 = std::min(gray.cols, cx + r);
    const int y2 = std::min(gray.rows, cy + r);

    if (x2 - x < 40 || y2 - y < 40)
        return false;

    cv::Mat roi = gray(cv::Rect(x, y, x2 - x, y2 - y));
    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 1.4);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
                     1.2, 24.0, 70.0, 17.0,
                     MIN_PUCK_RADIUS, MAX_PUCK_RADIUS);

    double bestScore = -1;
    cv::Point2f bestCenter;
    double bestRadius = EXPECTED_RADIUS;

    for (const auto &c : circles)
    {
        cv::Point2f center(static_cast<float>(x + c[0]),
                           static_cast<float>(y + c[1]));
        const double predictionDistance = distanceBetween(center, track.predicted);
        const double staticDistance = nearestStaticDistance(center, statics);
        const double motionValue = motionFraction(motion, center, c[2]);

        if (staticDistance < 28.0 && motionValue < 0.12)
            continue;

        double score = 2.0 * std::exp(-predictionDistance / 70.0) +
                       2.0 * std::clamp(motionValue / 0.25, 0.0, 1.0);

        if (staticDistance < 45.0)
            score -= (45.0 - staticDistance) / 35.0;

        if (score > bestScore)
        {
            bestScore = score;
            bestCenter = center;
            bestRadius = c[2];
        }
    }

    if (bestScore < 1.1)
        return false;

    Detection d;
    d.center = bestCenter;
    d.radius = bestRadius;
    d.area = CV_PI * bestRadius * bestRadius;
    d.quality = 0.45;

    updateMoving(track, d, gray, frame, 0.42, false);
    return true;
}

void drawStatic(cv::Mat &frame, const StaticPuck &puck, int id)
{
    const cv::Point center(static_cast<int>(std::round(puck.center.x)),
                           static_cast<int>(std::round(puck.center.y)));
    const int radius = std::max(12, static_cast<int>(std::round(puck.radius + 7)));

    cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2);
    cv::circle(frame, center, 4, cv::Scalar(0, 255, 255), -1);

    std::ostringstream text;
    text << "Шайба #" << id << " СТАТИЧНАЯ";
    cv::putText(frame, text.str(), center + cv::Point(10, -radius),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

    std::ostringstream pos;
    pos << std::fixed << std::setprecision(1)
        << "(" << puck.center.x << ", " << puck.center.y << ")";
    cv::putText(frame, pos.str(), center + cv::Point(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
}

void drawMoving(cv::Mat &frame, const MovingTrack &track, int id)
{
    if (!track.initialized)
        return;

    const cv::Point center(static_cast<int>(std::round(track.position.x)),
                           static_cast<int>(std::round(track.position.y)));
    const int radius = std::max(13, static_cast<int>(std::round(track.radius + 7)));

    cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2);
    cv::circle(frame, center, 4, cv::Scalar(0, 0, 255), -1);

    const cv::Point2f pred = track.predicted;
    cv::circle(frame, cv::Point(static_cast<int>(pred.x), static_cast<int>(pred.y)),
               5, cv::Scalar(255, 0, 255), 1);

    cv::arrowedLine(frame, center,
                    cv::Point(static_cast<int>(center.x + track.velocity.x * 3.0),
                              static_cast<int>(center.y + track.velocity.y * 3.0)),
                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    std::ostringstream name;
    name << "Шайба #" << id << " ДВИЖЕНИЕ";
    cv::putText(frame, name.str(), center + cv::Point(10, -radius),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

    std::ostringstream info;
    info << std::fixed << std::setprecision(1)
         << "(" << track.position.x << ", " << track.position.y << ")";
    cv::putText(frame, info.str(), center + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 255), 1);
}

// ---------- VideoReader ----------
VideoReader::VideoReader(const std::string &path,
                         BlockingQueue<FramePacket> &output,
                         std::atomic<bool> &stop,
                         std::atomic<bool> &paused,
                         double targetFps)
    : path_(path), output_(output), stop_(stop), paused_(paused), targetFps_(targetFps) {}

void VideoReader::run()
{
    cv::VideoCapture cap(path_);
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open video: " << path_ << "\n";
        output_.close();
        return;
    }

    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0)
        videoFps = 30.0;

    double effectiveFps = (targetFps_ > 0) ? targetFps_ : videoFps;
    int delayMs = static_cast<int>(1000.0 / effectiveFps);

    int frameNumber = 0;
    cv::Mat frame;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!stop_.load())
    {
        while (paused_.load() && !stop_.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (!cap.read(frame))
            break;

        FramePacket packet;
        packet.frame = frame.clone();
        packet.number = frameNumber++;
        packet.fps = effectiveFps;
        if (!output_.push(std::move(packet)))
            break;

        if (targetFps_ > 0)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime);
            int sleepMs = delayMs - static_cast<int>(elapsed.count());
            if (sleepMs > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            lastTime = std::chrono::high_resolution_clock::now();
        }
    }
    cap.release();
    output_.close();
}

// ---------- TrackerWorker ----------
TrackerWorker::TrackerWorker(BlockingQueue<FramePacket> &input,
                             BlockingQueue<ResultPacket> &output,
                             std::atomic<bool> &stop,
                             std::atomic<bool> &paused)
    : input_(input), output_(output), stop_(stop), paused_(paused) {}

void TrackerWorker::run()
{
    std::vector<StaticCluster> clusters;
    std::vector<StaticPuck> statics;
    MovingTrack moving;
    cv::Mat previousGray;
    bool calibrated = false;
    int calibrationFrames = 0;

    FramePacket packet;
    while (!stop_.load() && input_.pop(packet))
    {
        while (paused_.load() && !stop_.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        cv::Mat gray;
        if (!prepareGray(packet.frame, gray))
            continue;

        cv::Mat binary;
        makeBinary(gray, binary);

        cv::Mat hsv;
        if (packet.frame.channels() == 3)
            cv::cvtColor(packet.frame, hsv, cv::COLOR_BGR2HSV);
        else if (packet.frame.channels() == 4)
        {
            cv::Mat bgr;
            cv::cvtColor(packet.frame, bgr, cv::COLOR_BGRA2BGR);
            cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        }
        else
        {
            cv::Mat bgr;
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        }

        cv::Mat motion;
        makeMotionMask(previousGray, gray, motion);

        std::vector<Detection> contourDetections = findContours(binary, gray, hsv);
        std::vector<Detection> houghDetections;
        if (contourDetections.size() < 2 || packet.number % 3 == 0)
            houghDetections = findHough(gray, hsv);

        std::vector<Detection> detections = mergeDetections(contourDetections, houghDetections);

        if (!calibrated)
        {
            updateStaticClusters(clusters, detections);
            calibrationFrames++;

            if (calibrationFrames >= CALIBRATION_FRAMES)
            {
                statics = buildStaticPucks(clusters, calibrationFrames);
                calibrated = !statics.empty();
                std::cout << "Калибровка завершена: статических шайб = " << statics.size() << "\n";
                for (size_t i = 0; i < statics.size(); ++i)
                    std::cout << "  static #" << i + 1
                              << " = (" << statics[i].center.x
                              << ", " << statics[i].center.y << ")\n";
            }

            if (!calibrated)
            {
                ResultPacket result;
                result.frame = packet.frame.clone();
                result.number = packet.number;
                result.calibrated = false;
                cv::putText(result.frame, "Калибровка...",
                            cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.7, cv::Scalar(0, 255, 255), 2);
                output_.push(std::move(result));
                gray.copyTo(previousGray);
                continue;
            }
        }

        if (!moving.initialized)
        {
            int best = -1;
            double bestScore = -1e9;

            for (size_t i = 0; i < detections.size(); ++i)
            {
                const Detection &d = detections[i];
                const double md = motionFraction(motion, d.center, d.radius);
                const double sd = nearestStaticDistance(d.center, statics);

                double score = 2.0 * d.quality +
                               2.2 * std::clamp(md / 0.25, 0.0, 1.0);
                if (sd < 45.0)
                    score -= (45.0 - sd) / 30.0;
                if (sd < 20.0 && md < 0.08)
                    score -= 2.0;

                if (score > bestScore)
                {
                    bestScore = score;
                    best = static_cast<int>(i);
                }
            }

            if (best >= 0 && bestScore >= 1.45)
            {
                const Detection &d = detections[best];
                const double md = motionFraction(motion, d.center, d.radius);
                const double sd = nearestStaticDistance(d.center, statics);
                if (md >= 0.06 || sd > 55.0)
                {
                    updateMoving(moving, d, gray, packet.number,
                                 std::clamp(0.45 + d.quality * 0.4 + md * 0.4, 0.0, 1.0),
                                 true);
                }
            }
        }
        else
        {
            const int best = chooseMovingDetection(detections, motion, moving, statics);
            bool found = false;

            if (best >= 0)
            {
                const Detection &d = detections[best];
                const double md = motionFraction(motion, d.center, d.radius);
                const double pd = distanceBetween(d.center, moving.predicted);
                const double sd = nearestStaticDistance(d.center, statics);

                double confidence = 0.45 +
                                    0.30 * d.quality +
                                    0.20 * std::clamp(md / 0.25, 0.0, 1.0) +
                                    0.15 * std::exp(-pd / 80.0);
                if (sd < 35.0)
                    confidence += 0.08;

                updateMoving(moving, d, gray, packet.number,
                             std::clamp(confidence, 0.0, 1.0),
                             true);
                found = true;
            }

            if (!found && recoverByTemplate(gray, motion, moving, statics, packet.number))
                found = true;

            if (!found && recoverByLocalHough(gray, motion, moving, statics, packet.number))
                found = true;

            if (!found)
                missMoving(moving, packet.frame);

            if (moving.missed > MAX_MISSED_FRAMES)
            {
                moving.initialized = false;
                moving.puckTemplate.release();
                moving.history.clear();
                moving.confidence = 0;
                moving.missed = 0;
            }
        }

        // Формируем результат и отправляем
        ResultPacket result;
        result.frame = packet.frame.clone();
        result.number = packet.number;
        result.fps = packet.fps;
        result.calibrationFrames = calibrationFrames;
        result.totalFrames = packet.number + 1;
        result.speed = moving.initialized ? magnitude(moving.velocity) : 0.0;
        result.trackedObjects = moving.initialized ? 1 : 0;
        result.statics = statics;
        result.moving = moving;
        result.calibrated = calibrated;

        for (size_t i = 0; i < statics.size(); ++i)
            drawStatic(result.frame, statics[i], static_cast<int>(i + 1));

        if (moving.initialized)
            drawMoving(result.frame, moving, static_cast<int>(statics.size() + 1));

        std::ostringstream info;
        info << "K: " << calibrationFrames << "/" << CALIBRATION_FRAMES
             << " | FPS: " << std::fixed << std::setprecision(1) << result.fps
             << " | SPEED: " << std::fixed << std::setprecision(1) << result.speed;
        cv::putText(result.frame, info.str(),
                    cv::Point(15, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1);

        output_.push(std::move(result));
        gray.copyTo(previousGray);
    }

    output_.close();
}