#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


int main()
{
    Mat img1 = imread("Paris1.jpg");
    Mat img2 = imread("Paris.jpg");

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    Ptr<SIFT> detector = SIFT::create(100,3,0.03,10,1.2);
    std::vector<KeyPoint> keypoints1, keypoints2;

    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    //-- Step 2: Matching descriptor vectors with a brute force matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2,true);
    std::vector< DMatch > matches;
    matcher->match(descriptors1, descriptors2, matches);
    //-- Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    imshow("Matches", img_matches);
    waitKey(0);
    return 0;
}