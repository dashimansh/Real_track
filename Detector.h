#pragma once
#include <opencv2/opencv.hpp>

struct DetectionResult
{
    bool      bFound = false;
    cv::Rect  BoundingBox;
    cv::Point Center;
    double    Area = 0.0;
    float     Confidence = 0.f;
};

class Detector
{
public:
    Detector();

    DetectionResult Detect(
        const cv::Mat& Frame);

    // Main auto ROI — tries all methods
    cv::Rect GetAutoROI(
        const cv::Mat& Frame,
        const cv::Point& ClickPoint,
        int DefaultSize = 80);

    cv::Rect SelectROI(cv::Mat& Frame);

    void SetColorRange(
        const cv::Scalar& Lower1,
        const cv::Scalar& Upper1,
        const cv::Scalar& Lower2 =
            cv::Scalar(-1, -1, -1),
        const cv::Scalar& Upper2 =
            cv::Scalar(-1, -1, -1));

    void AdaptColorRange(
        const cv::Mat& Frame,
        const cv::Rect& ObjectBox);

    void Draw(cv::Mat& Frame,
        const DetectionResult& Result);

    cv::Mat GetMask() const
    {
        return Mask;
    }

private:
    cv::Scalar ColorLower1;
    cv::Scalar ColorUpper1;
    cv::Scalar ColorLower2;
    cv::Scalar ColorUpper2;
    bool       bHasTwoRanges = false;
    float      AdaptRate = 0.05f;

    cv::Mat HSVFrame;
    cv::Mat Mask;
    cv::Mat Mask2;

    void BuildMask(const cv::Mat& Frame);
    DetectionResult FindLargestBlob();

    // Method 1 — GrabCut
    // Most accurate for any object
    cv::Rect GetROIByGrabCut(
        const cv::Mat& Frame,
        const cv::Point& ClickPoint,
        int InitSize = 160);

    // Method 2 — Edge detection
    // Works for textured objects
    cv::Rect GetROIByEdges(
        const cv::Mat& Frame,
        const cv::Point& ClickPoint,
        int SearchRadius = 120);

    // Method 3 — Color similarity
    // Fallback method
    cv::Rect GetROIByColor(
        const cv::Mat& Frame,
        const cv::Point& ClickPoint,
        int SearchRadius = 120);
};
