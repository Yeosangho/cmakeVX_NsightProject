#include "homography_smoother.hpp"
HomographySmoother::HomographySmoother(int smoothing_window_size):
    smoothing_window_size_(smoothing_window_size)
{
    for (int i = 0; i < smoothing_window_size; i++)
        homography_matrices_.push_back(cv::Matx33f::eye());
    float sigma = smoothing_window_size_ * 0.7;
    for (int i = -smoothing_window_size_; i <= smoothing_window_size_; ++i)
        gaussWeights_.push_back( exp(-i * i / (2.f * sigma * sigma)) );
    //normalize weights
    float sum = 0;
    int size = gaussWeights_.size();
    for (int i = 0; i < size; i++)
        sum += gaussWeights_[i];
    float scaler = 1.f / sum;
    for (int i = 0; i < size; i++)
        gaussWeights_[i] *= scaler;
}
void HomographySmoother::push(const cv::Matx33f &transform)
{
    if (homography_matrices_.size() >= static_cast<uint>(2 * smoothing_window_size_))
    {
        std::rotate(homography_matrices_.begin(), homography_matrices_.begin() + 1, homography_matrices_.end());
        homography_matrices_.pop_back();
    }
    homography_matrices_.push_back(transform);
}
bool HomographySmoother::getSmoothedHomography(cv::Matx33f &transform)
{
    int size = homography_matrices_.size();
    if (size < 2 * smoothing_window_size_)
        return false;
    assert(size % 2 == 0);
    int idx = size / 2;
    transform = cv::Matx33f::zeros();
    for (int i = 0; i <= 2 * smoothing_window_size_; ++i)
    {
        transform += gaussWeights_[i] * getTransformation(idx, i);
    }
    return true;
}
cv::Matx33f HomographySmoother::getTransformation(int from, int to)
{
    cv::Matx33f transformation = cv::Matx33f::eye();
    if (to > from)
    {
        for (int i = from; i < to; ++i)
        {
            transformation = transformation * homography_matrices_[i];
        }
    }
    else if (to < from)
    {
        for (int i = to; i < from; ++i)
        {
            transformation = transformation * homography_matrices_[i];
        }
        transformation = transformation.inv();
    }
    return transformation;
}
