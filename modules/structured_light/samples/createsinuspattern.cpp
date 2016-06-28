/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/structured_light.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace cv;
using namespace std;

static const char* keys =
{
    "{@width | | Projector width}"
    "{@height | | Projector height}"
    "{@periods | | Number of periods}"
    "{@setMarkers | | Patterns with or without markers}"
    "{@horizontal | | Patterns are horizontal}"
    "{@methodId | | Method to be used}"
    "{@outputPath | | Path to save patterns}"
};
static void help()
{
    cout << "\nThis example generates sinusoidal patterns" << endl;
    cout << "To call: ./example_structured_light_createsinuspattern <width> <height>"
            " <number_of_period> <set_marker>(bool) <horizontal_patterns>(bool) <method_id>"
            " <output_path>(optional)"<< endl;
}
int main(int argc, char **argv)
{
    if( argc < 2 )
    {
        help();
        return -1;
    }
    structured_light::SinusoidalPattern::Params params;

    // Retrieve parameters written in the command line
    CommandLineParser parser(argc, argv, keys);
    params.width = parser.get<int>(0);
    params.height = parser.get<int>(1);
    params.nbrOfPeriods = parser.get<int>(2);
    params.setMarkers = parser.get<bool>(3);
    params.horizontal = parser.get<bool>(4);
    params.methodId = parser.get<int>(5);
    String outputPath = parser.get<String>(6);

    params.shiftValue = static_cast<float>(2 * CV_PI / 3);
    params.nbrOfPixelsBetweenMarkers = 70;

    Ptr<structured_light::SinusoidalPattern> sinus = structured_light::SinusoidalPattern::create(params);

    vector<Mat> patterns;
    Mat shadowMask;

    // Generate sinusoidal patterns
    sinus->generate(patterns);
    Mat phaseMap, phaseMap8;
    // Compute wrapped phase map
    sinus->computePhaseMap(patterns, phaseMap, shadowMask);
    phaseMap.convertTo(phaseMap8, CV_8U, 255, 128);
    imshow("Wrapped phase map", phaseMap8);

    // Save patterns
    if( !outputPath.empty() )
    {
        for( int i = 0; i < 3; ++ i )
        {
            ostringstream name;
            name << i + 1;
            imwrite(outputPath + name.str() + ".png", patterns[i]);
        }
    }
    bool loop = true;
    while( loop )
    {
        int key = waitKey(0);
        if( key == 27 )
        {
            loop = false;
        }
    }

    return 0;
}