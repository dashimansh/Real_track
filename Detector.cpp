#include "Detector.h"
#include "Config.h"
#include <iostream>

Detector::Detector()
{
    ColorLower1 = cv::Scalar(
        RED_LOWER1_H, RED_LOWER1_S,
        RED_LOWER1_V);
    ColorUpper1 = cv::Scalar(
        RED_UPPER1_H, RED_UPPER1_S,
        RED_UPPER1_V);
    ColorLower2 = cv::Scalar(
        RED_LOWER2_H, RED_LOWER2_S,
        RED_LOWER2_V);
    ColorUpper2 = cv::Scalar(
        RED_UPPER2_H, RED_UPPER2_S,
        RED_UPPER2_V);
    bHasTwoRanges = true;
}

void Detector::SetColorRange(
    const cv::Scalar& Lower1,
    const cv::Scalar& Upper1,
    const cv::Scalar& Lower2,
    const cv::Scalar& Upper2)
{
    ColorLower1 = Lower1;
    ColorUpper1 = Upper1;
    if (Lower2[0] >= 0)
    {
        ColorLower2 = Lower2;
        ColorUpper2 = Upper2;
        bHasTwoRanges = true;
    }
    else bHasTwoRanges = false;
}

void Detector::BuildMask(
    const cv::Mat& Frame)
{
    cv::cvtColor(Frame, HSVFrame,
        cv::COLOR_BGR2HSV);
    cv::inRange(HSVFrame,
        ColorLower1, ColorUpper1, Mask);
    if (bHasTwoRanges)
    {
        cv::inRange(HSVFrame,
            ColorLower2, ColorUpper2,
            Mask2);
        cv::bitwise_or(Mask, Mask2, Mask);
    }
    cv::erode(Mask, Mask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(Mask, Mask,
        cv::Mat(), cv::Point(-1,-1), 3);
}

DetectionResult Detector::FindLargestBlob()
{
    DetectionResult Result;
    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(Mask, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty()) return Result;

    double LargestArea = 0;
    int    LargestIdx  = -1;
    for (int i = 0;
        i < (int)Contours.size(); i++)
    {
        double Area =
            cv::contourArea(Contours[i]);
        if (Area > LargestArea)
        {
            LargestArea = Area;
            LargestIdx  = i;
        }
    }

    if (LargestArea < MIN_OBJECT_AREA)
        return Result;

    Result.bFound      = true;
    Result.Area        = LargestArea;
    Result.BoundingBox =
        cv::boundingRect(
            Contours[LargestIdx]);
    Result.Center = cv::Point(
        Result.BoundingBox.x +
        Result.BoundingBox.width  / 2,
        Result.BoundingBox.y +
        Result.BoundingBox.height / 2);

    float BoxArea = (float)(
        Result.BoundingBox.width *
        Result.BoundingBox.height);
    Result.Confidence = std::min(1.f,
        (float)(LargestArea / BoxArea));
    return Result;
}

DetectionResult Detector::Detect(
    const cv::Mat& Frame)
{
    if (Frame.empty())
        return DetectionResult();
    BuildMask(Frame);
    return FindLargestBlob();
}

// ============================================
// METHOD 1 — GRABCUT
// Most accurate ROI detection
// Separates foreground from background
// Works for ANY object!
// ============================================
cv::Rect Detector::GetROIByGrabCut(
    const cv::Mat& Frame,
    const cv::Point& ClickPoint,
    int InitSize)
{
    // Initial rectangle around
    // click point for GrabCut
    int Half = InitSize / 2;
    cv::Rect InitRect(
        std::max(0, ClickPoint.x - Half),
        std::max(0, ClickPoint.y - Half),
        std::min(InitSize,
            Frame.cols - 
            std::max(0,
                ClickPoint.x - Half)),
        std::min(InitSize,
            Frame.rows - 
            std::max(0,
                ClickPoint.y - Half)));

    if (InitRect.width  < 10 ||
        InitRect.height < 10)
        return cv::Rect();

    // GrabCut needs at least
    // 3x3 border around rect
    InitRect.x = std::max(3,
        InitRect.x);
    InitRect.y = std::max(3,
        InitRect.y);
    InitRect.width = std::min(
        InitRect.width,
        Frame.cols - InitRect.x - 3);
    InitRect.height = std::min(
        InitRect.height,
        Frame.rows - InitRect.y - 3);

    if (InitRect.width  < 10 ||
        InitRect.height < 10)
        return cv::Rect();

    try
    {
        cv::Mat BGModel, FGModel;
        cv::Mat GrabMask(
            Frame.rows, Frame.cols,
            CV_8UC1,
            cv::Scalar(cv::GC_BGD));

        // Run GrabCut — separates
        // foreground from background
        cv::grabCut(Frame,
            GrabMask,
            InitRect,
            BGModel, FGModel,
            3,  // iterations
            cv::GC_INIT_WITH_RECT);

        // Create foreground mask
        cv::Mat FGMask =
            (GrabMask == cv::GC_FGD) |
            (GrabMask == cv::GC_PR_FGD);

        // Clean up mask
        cv::erode(FGMask, FGMask,
            cv::Mat(),
            cv::Point(-1,-1), 2);
        cv::dilate(FGMask, FGMask,
            cv::Mat(),
            cv::Point(-1,-1), 3);

        // Find contours in FG mask
        std::vector<
            std::vector<cv::Point>>
            Contours;
        cv::findContours(FGMask,
            Contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        if (Contours.empty())
            return cv::Rect();

        // Find contour nearest to
        // click point
        double BestDist = 1e9;
        cv::Rect BestBox;
        bool bFound = false;

        for (auto& C : Contours)
        {
            if (cv::contourArea(C) < 200)
                continue;

            cv::Rect Box =
                cv::boundingRect(C);

            cv::Point BoxCenter(
                Box.x + Box.width  / 2,
                Box.y + Box.height / 2);

            double Dist = cv::norm(
                BoxCenter - ClickPoint);

            if (Dist < BestDist)
            {
                BestDist = Dist;
                BestBox  = Box;
                bFound   = true;
            }
        }

        if (!bFound) return cv::Rect();

        // Add small padding
        int Pad = 8;
        BestBox.x -= Pad;
        BestBox.y -= Pad;
        BestBox.width  += Pad * 2;
        BestBox.height += Pad * 2;
        BestBox &= cv::Rect(0, 0,
            Frame.cols, Frame.rows);

        std::cout
            << "GrabCut ROI: "
            << BestBox.width
            << "x"
            << BestBox.height
            << "\n";

        return BestBox;
    }
    catch (...)
    {
        std::cout
            << "GrabCut failed,"
            << " trying edge method\n";
        return cv::Rect();
    }
}

// ============================================
// METHOD 2 — EDGE DETECTION
// Uses Canny edges to find object
// boundary regardless of color
// Works for textured objects!
// ============================================
cv::Rect Detector::GetROIByEdges(
    const cv::Mat& Frame,
    const cv::Point& ClickPoint,
    int SearchRadius)
{
    // Work in a search region
    // around click point
    int X1 = std::max(0,
        ClickPoint.x - SearchRadius);
    int Y1 = std::max(0,
        ClickPoint.y - SearchRadius);
    int X2 = std::min(Frame.cols - 1,
        ClickPoint.x + SearchRadius);
    int Y2 = std::min(Frame.rows - 1,
        ClickPoint.y + SearchRadius);

    cv::Rect SearchRect(
        X1, Y1,
        X2 - X1, Y2 - Y1);

    if (SearchRect.width  < 20 ||
        SearchRect.height < 20)
        return cv::Rect();

    cv::Mat SearchRegion =
        Frame(SearchRect).clone();

    // Convert to grayscale
    cv::Mat Gray;
    cv::cvtColor(SearchRegion, Gray,
        cv::COLOR_BGR2GRAY);

    // Blur to reduce noise
    cv::GaussianBlur(Gray, Gray,
        cv::Size(5, 5), 1.5);

    // Canny edge detection
    cv::Mat Edges;
    cv::Canny(Gray, Edges, 30, 90);

    // Dilate edges to close gaps
    cv::dilate(Edges, Edges,
        cv::Mat(),
        cv::Point(-1,-1), 3);

    // Find contours from edges
    std::vector<
        std::vector<cv::Point>>
        Contours;
    cv::findContours(Edges, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty())
        return cv::Rect();

    // Local click point in
    // search region coordinates
    cv::Point LocalClick(
        ClickPoint.x - X1,
        ClickPoint.y - Y1);

    // Find contour nearest to
    // click point with good size
    double BestScore = 1e9;
    cv::Rect BestBox;
    bool bFound = false;

    for (auto& C : Contours)
    {
        double Area = cv::contourArea(C);
        if (Area < 300) continue;

        cv::Rect Box =
            cv::boundingRect(C);

        cv::Point BoxCenter(
            Box.x + Box.width  / 2,
            Box.y + Box.height / 2);

        double Dist = cv::norm(
            BoxCenter - LocalClick);

        // Score combines distance
        // and size — prefer large
        // contours near click point
        double Score =
            Dist / std::sqrt(Area);

        if (Score < BestScore)
        {
            BestScore = Score;
            BestBox   = Box;
            bFound    = true;
        }
    }

    if (!bFound) return cv::Rect();

    // Convert back to frame coordinates
    BestBox.x += X1;
    BestBox.y += Y1;

    // Add padding
    int Pad = 10;
    BestBox.x -= Pad;
    BestBox.y -= Pad;
    BestBox.width  += Pad * 2;
    BestBox.height += Pad * 2;
    BestBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    std::cout
        << "Edge ROI: "
        << BestBox.width
        << "x"
        << BestBox.height
        << "\n";

    return BestBox;
}

// ============================================
// METHOD 3 — COLOR SIMILARITY
// Original method — kept as fallback
// ============================================
cv::Rect Detector::GetROIByColor(
    const cv::Mat& Frame,
    const cv::Point& ClickPoint,
    int SearchRadius)
{
    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Vec3b ClickHSV =
        HSV.at<cv::Vec3b>(
            ClickPoint.y,
            ClickPoint.x);

    int HTol = 15;
    int STol = 60;
    int VTol = 60;

    int X1 = std::max(0,
        ClickPoint.x - SearchRadius);
    int Y1 = std::max(0,
        ClickPoint.y - SearchRadius);
    int X2 = std::min(Frame.cols - 1,
        ClickPoint.x + SearchRadius);
    int Y2 = std::min(Frame.rows - 1,
        ClickPoint.y + SearchRadius);

    cv::Mat SimilarMask =
        cv::Mat::zeros(
            Frame.rows, Frame.cols,
            CV_8UC1);

    for (int Y = Y1; Y <= Y2; Y++)
    {
        for (int X = X1; X <= X2; X++)
        {
            cv::Vec3b PixelHSV =
                HSV.at<cv::Vec3b>(Y, X);

            int HDiff = std::abs(
                (int)PixelHSV[0] -
                (int)ClickHSV[0]);
            int SDiff = std::abs(
                (int)PixelHSV[1] -
                (int)ClickHSV[1]);
            int VDiff = std::abs(
                (int)PixelHSV[2] -
                (int)ClickHSV[2]);

            if (HDiff > 90)
                HDiff = 180 - HDiff;

            if (HDiff <= HTol &&
                SDiff <= STol &&
                VDiff <= VTol)
                SimilarMask.at<uchar>(
                    Y, X) = 255;
        }
    }

    cv::erode(SimilarMask, SimilarMask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(SimilarMask, SimilarMask,
        cv::Mat(), cv::Point(-1,-1), 4);

    std::vector<
        std::vector<cv::Point>>
        Contours;
    cv::findContours(SimilarMask,
        Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty())
        return cv::Rect();

    for (auto& C : Contours)
    {
        cv::Rect Box =
            cv::boundingRect(C);
        if (Box.contains(ClickPoint) &&
            Box.area() > 400)
        {
            int Pad = 10;
            Box.x -= Pad;
            Box.y -= Pad;
            Box.width  += Pad * 2;
            Box.height += Pad * 2;
            Box &= cv::Rect(0, 0,
                Frame.cols, Frame.rows);

            std::cout
                << "Color ROI: "
                << Box.width
                << "x"
                << Box.height
                << "\n";
            return Box;
        }
    }
    return cv::Rect();
}

// ============================================
// MAIN AUTO ROI FUNCTION
// Tries all 3 methods in order:
// 1. GrabCut  — most accurate
// 2. Edges    — works for any texture
// 3. Color    — fallback
// 4. Default  — last resort fixed size
// ============================================
cv::Rect Detector::GetAutoROI(
    const cv::Mat& Frame,
    const cv::Point& ClickPoint,
    int DefaultSize)
{
    // Validate click point
    if (ClickPoint.x < 5 ||
        ClickPoint.y < 5 ||
        ClickPoint.x >= Frame.cols - 5 ||
        ClickPoint.y >= Frame.rows - 5)
    {
        int Half = DefaultSize / 2;
        return cv::Rect(
            ClickPoint.x - Half,
            ClickPoint.y - Half,
            DefaultSize,
            DefaultSize) &
            cv::Rect(0, 0,
                Frame.cols, Frame.rows);
    }

    // ---- METHOD 1: GrabCut ----
    std::cout
        << "Trying GrabCut...\n";
    cv::Rect GrabResult =
        GetROIByGrabCut(
            Frame, ClickPoint,
            DefaultSize * 3);

    if (GrabResult.area() > 400)
    {
        std::cout
            << "GrabCut SUCCESS!\n";
        return GrabResult;
    }

    // ---- METHOD 2: Edge Detection ----
    std::cout
        << "Trying Edge detection...\n";
    cv::Rect EdgeResult =
        GetROIByEdges(
            Frame, ClickPoint,
            DefaultSize * 2);

    if (EdgeResult.area() > 400)
    {
        std::cout
            << "Edge detection SUCCESS!\n";
        return EdgeResult;
    }

    // ---- METHOD 3: Color Similarity ----
    std::cout
        << "Trying Color similarity...\n";
    cv::Rect ColorResult =
        GetROIByColor(
            Frame, ClickPoint,
            DefaultSize * 2);

    if (ColorResult.area() > 400)
    {
        std::cout
            << "Color similarity SUCCESS!\n";
        return ColorResult;
    }

    // ---- METHOD 4: Default size ----
    std::cout
        << "Using default size ROI\n";
    int Half = DefaultSize / 2;
    cv::Rect DefaultBox(
        ClickPoint.x - Half,
        ClickPoint.y - Half,
        DefaultSize, DefaultSize);

    DefaultBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);
    return DefaultBox;
}

void Detector::AdaptColorRange(
    const cv::Mat& Frame,
    const cv::Rect& ObjectBox)
{
    cv::Rect SafeBox = ObjectBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0) return;

    cv::Mat ObjectHSV;
    cv::cvtColor(Frame(SafeBox),
        ObjectHSV,
        cv::COLOR_BGR2HSV);

    cv::Scalar Mean, StdDev;
    cv::meanStdDev(ObjectHSV,
        Mean, StdDev);

    float Margin = 15.f;
    cv::Scalar NewLower(
        std::max(0.0, Mean[0] - Margin),
        std::max(0.0, Mean[1] - 40.0),
        std::max(0.0, Mean[2] - 40.0));
    cv::Scalar NewUpper(
        std::min(180.0, Mean[0] + Margin),
        255.0, 255.0);

    for (int i = 0; i < 3; i++)
    {
        ColorLower1[i] =
            ColorLower1[i] *
            (1 - AdaptRate) +
            NewLower[i] * AdaptRate;
        ColorUpper1[i] =
            ColorUpper1[i] *
            (1 - AdaptRate) +
            NewUpper[i] * AdaptRate;
    }
    bHasTwoRanges = false;
}

cv::Rect Detector::SelectROI(
    cv::Mat& Frame)
{
    cv::Rect ROI = cv::selectROI(
        "Select Object - press Enter",
        Frame, false, false);
    cv::destroyWindow(
        "Select Object - press Enter");
    return ROI;
}

void Detector::Draw(
    cv::Mat& Frame,
    const DetectionResult& Result)
{
    if (!Result.bFound) return;
    cv::rectangle(Frame,
        Result.BoundingBox,
        cv::Scalar(0, 165, 255), 2);
    cv::putText(Frame,
        "DETECTED " +
        std::to_string(
            (int)(Result.Confidence * 100))
        + "%",
        cv::Point(
            Result.BoundingBox.x,
            Result.BoundingBox.y - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 165, 255), 2);
}
