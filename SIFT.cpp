#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


int main()
{
    Mat img1 = imread(("BD.jpg"),IMREAD_GRAYSCALE);
    Mat img2 = imread(("BP2.jpg"),IMREAD_GRAYSCALE);

    Mat mg1, mg2;
    resize(img1, mg1, Size(360, 240));
    resize(img2, mg2, Size(360, 240));

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    Ptr<SIFT> detector = SIFT::create(0,2,0.04,10,1.3);
    std::vector<KeyPoint> keypoints1, keypoints2;

    Mat descriptors1, descriptors2;
    detector->detectAndCompute(mg1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(mg2, noArray(), keypoints2, descriptors2);

    //-- Step 2: Matching descriptor vectors with a brute force matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    std::vector< DMatch > matches;
    matcher->match(descriptors1, descriptors2, matches);

    double max_dist = 0;
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }



    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);



    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;



    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance < 2.5 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }



    Mat img_matches;
    drawMatches(mg1, keypoints1, mg2, keypoints2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;



    for (int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }



    Mat H = findHomography(obj, scene, RANSAC);



    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(mg1.cols, 0);
    obj_corners[2] = Point(mg1.cols, mg1.rows);
    obj_corners[3] = Point(0, mg1.rows);
    std::vector<Point2f> scene_corners(4);



    perspectiveTransform(obj_corners, scene_corners, H);



    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0] + Point2f(mg1.cols, 0), scene_corners[1] + Point2f(mg1.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f(mg1.cols, 0), scene_corners[2] + Point2f(mg1.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f(mg1.cols, 0), scene_corners[3] + Point2f(mg1.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f(mg1.cols, 0), scene_corners[0] + Point2f(mg1.cols, 0), Scalar(0, 255, 0), 4);



    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches);



    waitKey(0);
    return 0;
}
