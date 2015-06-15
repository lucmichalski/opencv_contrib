/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2015
 * Seungryung Kim
 * web : http://seungryong.github.io/DASC/
 * email : srkim89 at yonsei dot ac dot kr
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/*
 "DASC: Dense Adaptive Self-Correlation Descriptor for Multi-Modal and
 Multi-Spectral Correspondence", Seungryong Kim, Dongbo Min, Bumsub Ham,
 Seungchul Ryu, Minh N. Do, Kwanghoon Sohn; The IEEE Conference on Computer
 Vision and Pattern Recognition (CVPR), 2015, pp. 2103-2112

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>
 */

#include "precomp.hpp"

#include <fstream>
#include <stdlib.h>

namespace cv
{
namespace xfeatures2d
{

/*
 !DASC implementation
 */
class DASC_Impl : public DASC
{

public:

    /** Constructor
      @param n_half
      @param epsilon
      @param down_size
     */
    static Ptr<DASC> create( int n_half = 2, float epsilon = 0.08, int down_size = 2 );

    explicit DASC_Impl( int n_half = 2, float epsilon = 0.08, int down_size = 2 );

    virtual ~DASC_Impl();

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const {
        // +1 is for center pixel
        return 128;
    };

    /** returns the descriptor type */
    virtual int descriptorType() const { return CV_32F; }

    /** returns the default norm type */
    virtual int defaultNorm() const { return NORM_L2; }


    /** @overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, OutputArray descriptors );

protected:

    int m_n_half;

    float m_epsilon;

    int m_down_size;


    Rect m_roi;

    Mat m_image;

private:

    Mat m_rp1;
    Mat m_rp2;


    Mat m_glob_I_sub;
    Mat m_glob_I_var;
    Mat m_glob_I_mean;
    Mat m_glob_II_mean;


    // image set image as working
    inline void set_image( InputArray image );

    // computes the descriptors for every pixel in the image.
    inline void compute_descriptors( Mat* m_dense_descriptors );

    // set sampling parameters
    inline void setSamplingPoints();

    inline void boxfilter( Mat& img, Mat& img_out, Mat& integral_img );

    inline void guidedfilter_precompute();

    inline void guidedfilter_runfilter( Mat& I, Mat& p, Mat& I_out );

}; // END DASC_Impl CLASS



// -------------------------------------------------
/* DASC interface implementation */

inline void DASC_Impl::set_image( InputArray _image )
{
    // fetch new image
    Mat image = _image.getMat();
    // image cannot be empty
    CV_Assert( ! image.empty() );
    // clone image for conversion
    if ( image.depth() != CV_32F ) {

      m_image = image.clone();
      // convert to gray inplace
      if( m_image.channels() > 1 )
          cvtColor( m_image, m_image, COLOR_BGR2GRAY );
      // convert and normalize
      m_image.convertTo( m_image, CV_32F );
      m_image /= 255.0f;
    } else
      // use original user supplied CV_32F image
      // should be a normalized one (cannot check)
      m_image = image;

    GaussianBlur( m_image, m_image, Size(7, 7), 1.0f, 1.0f, BORDER_REPLICATE );
    //resize( m_image, m_image, Size(0, 0), 0.75f, 0.75f, INTER_CUBIC ); // are we sure ?

}

inline void DASC_Impl::boxfilter( Mat& img, Mat& img_out, Mat& integral_img )
{
    // BOX kernel
    int kernel_size = (2 * m_n_half + 1);
    Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F );
    kernel /= (float) (kernel_size*kernel_size);

    integral( img, integral_img, CV_32F );
    filter2D( integral_img, img_out, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT );
}

inline void DASC_Impl::guidedfilter_precompute()
{

    Mat integral_img;

    int rows = m_image.rows;
    int cols = m_image.cols;
    int sub_rows = rows / m_down_size;
    int sub_cols = cols / m_down_size;

    // sub image
    resize( m_image, m_glob_I_sub, Size(sub_rows, sub_cols), 0, 0, INTER_NEAREST );

    boxfilter( m_glob_I_sub, m_glob_I_mean, integral_img );

    m_glob_II_mean = m_glob_I_sub.mul( m_glob_I_sub );

    boxfilter( m_glob_II_mean, m_glob_II_mean, integral_img );

    m_glob_I_var = m_glob_II_mean - m_glob_I_mean.mul(m_glob_I_mean) + m_epsilon;
}

inline void DASC_Impl::guidedfilter_runfilter( Mat& I, Mat& p, Mat& I_out )
{

    Mat integral_img;

    int rows = m_image.rows;
    int cols = m_image.cols;
    int sub_rows = rows / m_down_size;
    int sub_cols = cols / m_down_size;


    Mat p_sub, p_mean;
    resize( p, p_sub, Size(sub_rows, sub_cols), 0, 0, INTER_NEAREST );

    Mat Ip_mean = m_glob_I_sub.mul(p_sub);

    boxfilter( p_sub, p_mean, integral_img );
    boxfilter( Ip_mean, Ip_mean, integral_img );

    Mat a_mean = (Ip_mean - m_glob_I_mean.mul(p_mean)) / m_glob_I_var;
    Mat b_mean = p_mean - a_mean.mul(m_glob_I_mean);

    boxfilter( a_mean, a_mean, integral_img );
    boxfilter( b_mean, b_mean, integral_img );

    Mat a_mean_large, b_mean_large;
    resize( a_mean, a_mean_large, Size(rows, cols), 0, 0, INTER_LINEAR );
    resize( b_mean, b_mean_large, Size(rows, cols), 0, 0, INTER_LINEAR );

    I_out = a_mean_large.mul(I) + b_mean_large;

}


// Computes the descriptor by sampling convoluted orientation maps.
inline void DASC_Impl::compute_descriptors( Mat* descriptors )
{

    Mat I = m_image;
    Mat II = I.mul(I);
    Mat diff_rp = m_rp1 - m_rp2;

    guidedfilter_precompute();

    Mat I_adaptive_mean;
    guidedfilter_runfilter( I, I, I_adaptive_mean );

    Mat II_adaptive_mean;
    guidedfilter_runfilter( I, II, II_adaptive_mean );

    Mat J( I.rows, I.cols, CV_32F );
    Mat JJ( I.rows, I.cols, CV_32F );
    Mat IJ( I.rows, I.cols, CV_32F );

    int f_dim = m_rp1.cols;
    for (int s = 0; s<f_dim; s++)
    {

      int m = diff_rp.at<char>(0,s);
      int n = diff_rp.at<char>(1,s);

      for (int i = 0; i<I.rows; i++)
      {
        for (int j = 0; j<I.cols; j++)
        {
          if (i + m > -1 && i + m < I.rows && j + n > -1 && j + n < I.cols)
          {
            J.at<float>(i,j) = I.at<float>(i+m,j+n);
            float mul = J.at<float>(i,j) * J.at<float>(i,j);
            JJ.at<float>(i,j) = mul;
            IJ.at<float>(i,j) = mul;
          }
        }
      }

      Mat J_adaptive_mean;
      Mat IJ_adaptive_mean;
      Mat JJ_adaptive_mean;

      guidedfilter_runfilter( I, J, J_adaptive_mean );
      guidedfilter_runfilter( I, JJ, JJ_adaptive_mean );
      guidedfilter_runfilter( I, IJ, IJ_adaptive_mean );

      // descriptor @(X,Y) @dim = s
      for (int ii = 0; ii<I.rows; ii++)
      {
        for (int jj = 0; jj<I.cols; jj++)
        {

          int i1 = (int)(ii + m_rp1.at<char>(0,s));
          int j1 = (int)(jj + m_rp1.at<char>(1,s));

          if (i1 > 0 && i1 < I.rows && j1 > 0 && j1 < I.cols)
          {

            float num_corrSurf = IJ_adaptive_mean.at<float>(i1,j1)
                                - I_adaptive_mean.at<float>(i1,j1) * J_adaptive_mean.at<float>(i1,j1);

            float dem_left = II_adaptive_mean.at<float>(i1,j1)
                            - I_adaptive_mean.at<float>(i1,j1) * I_adaptive_mean.at<float>(i1,j1);

            float dem_right = JJ_adaptive_mean.at<float>(i1,j1)
                             - J_adaptive_mean.at<float>(i1,j1) * J_adaptive_mean.at<float>(i1,j1);

            float dem_corrSurf = sqrt( dem_left * dem_right );

            descriptors->at<float>(ii*I.cols+jj, s) = max( exp( -(1-num_corrSurf/dem_corrSurf)/0.5f ), 0.03f );

          }
        }
      }
    }
}

inline void DASC_Impl::setSamplingPoints()
{
    int rp1[] =
    {
       1,  4,  2,  7, -4,  2,  0, -5,  2,  4,  1,  8,  2, -2,  2,  7,
      -1,  7,  0,  4, -3,  1,  2,  4, -1,  0, -1, -6, -3,  6, -1,  9,
       0,  2,  2,  2, -5,  0,  1,  9, -4, -3,  0,  1,  2, -5, -8, -1,
      -1,  0,  0, -1, -1,  3,  1, -9,  4,  0, -8,  3, -2,  2, -1,  0,
       1,  4, -5,  7,  1,  0,  0,  2,  1,  1,  2, -1,  5,  2, -1, -8,
      -1,  3, -8, -5,  7,  0,  0,  0, -3, -2,  2,  5, -9,  0,  4,  2,
      -2, -8,  1, -3,  3,  1,  0, -9,  6, -4,  0, -1, -2, -2, -1,  8,
       0,  7,  1,  2, -2,  9,  4, -8,  4,  4,  0,  0,  1,  1,  8, -1,

      -2, -3,  1, -6, -3,  1,  9, -1, -1, -2,  2, -3, -1,  0, -9,  6,
       0, -6, -2, -2, -4,  0, -4, -2, -2, -1, -5,  7, -4,  7, -2,  0,
       9,  0,  1, -4, -8,  1,  0,  0,  2,  4, -2, -5,  4,  1,  4, -1,
      -2, -1,  9, -2, -2,  4,  1,  2, -2,  1, -3,  8, -9,  9,  0, -1,
      -1, -2, -1,  6,  1,  1, -1,  9, -2,  5, -1,  2,  8,  9, -1,  3,
       2, -8, -3, -1, -6,  1, -1, -1,  4, -1,  1,  1,  0, -9, -2,  1,
      -1,  3,  0,  8,  4, -5,  1,  0,  7, -2,  2, -2, -4, -9,  0,  3,
       1, -6,  2, -4,  0, -2,  2,  4,  2, -2,  0, -9, -2, -1,  3,  1
    };

    int rp2[] =
    {
       0,  2, -2,  7,  1,  0,  1,  4,  1, -8, -5,  1,  2, -2, -1, -2,
      -2,  4,  0,  1,  1,  0, -7,  5,  2,  8, -5, -2, -3,  2, -1,  8,
       0, -6,  1, -2,  1,  4,  8,  2, -4,  9,  0, -2,  2,  2,  2,  6,
      -1,  0, -1, -1, -5,  0, -1,  2, -2,  1, -1, -2,  0,  4,  2,  0,
      -2, -4, -8, -2, -1, -2,  1,  1, -1, -1,  0, -1, -1,  8,  3, -5,
      -2, -1,  1, -1, -8,  0,  1,  5,  3,  6,  0, -4,  0,  1, -2,  1,
       4,  2, -4,  8,  0, -4, -2,  0, -4, -2,  3,  1, -1, -1, -1, -1,
       8,  0,  2,  9,  2,  0, -2, -4,  0, -2,  4, -5, -5,  0, -2, -2,

       1, -9,  1, -6,  2,  2,  1,  3, -2, -3, -1,  1, -4, -1,  2,  0,
       1,  2,  2,  1,  2,  2,  6,  1, -1, -5,  1,  0,  8, -1,  1,  4,
      -1,  7,  5,  1,  0,  3, -5, -1, -2,  2,  2,  4, -4,  0,  0,  7,
      -2, -2, -1,  0, -1,  2, -5,  4, -4,  2,  1, -1, -1,  2, -1, -9,
       0,  2,  4,  9,  0, -2, -5, -1,  2, -1,  0,  2, -5, -4, -1, -1,
      -1,  0,  2,  5,  3,  1, -2,  8,  8,  7,  1,  2,  1,  0,  1,  2,
      -3,  0,  2,  4,  1,  3, -1,  2, -2, -4,  8,  1,  5,  2,  0,  0,
       3,  2, -9,  0, -1, -1,  0, -2,  2,  1, -2, -1,  0,  1,  0, -1
    };

    m_rp1 = Mat( 2, 128, CV_8S, rp1 );
    m_rp1 = Mat( 2, 128, CV_8S, rp2 );
}

// full scope
void DASC_Impl::compute( InputArray _image, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    set_image( _image );

    _descriptors.create( m_roi.width*m_roi.height, 128, CV_32F );

    Mat descriptors = _descriptors.getMat();

    // compute full desc
    compute_descriptors( &descriptors );

}

// constructor
DASC_Impl::DASC_Impl( int _n_half, float _epsilon, int _down_size )
          : m_n_half(_n_half), m_epsilon(_epsilon), m_down_size(_down_size)
{
}

// destructor
DASC_Impl::~DASC_Impl()
{
}

Ptr<DASC> DASC::create( int n_half, float epsilon, int down_size )
{
    return makePtr<DASC_Impl>(n_half, epsilon, down_size);
}



} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
