#ifndef _IMMEDIATE_MODE_STABILIZER_
#define _IMMEDIATE_MODE_STABILIZER_
#include <vector>
#include <queue>
#include "homography_smoother.hpp"
#include "NVX/nvx.h"
struct ImmediateModeStabilizerParams
{
    vx_size smoothing_window_size;
    vx_float32 fast_threshold;
    vx_int32 num_pyramid_levels;
    int opt_flow_win_size;
    float opt_flow_epsilon;
    int opt_flow_num_iterations;
    int opt_flow_use_initial_estimate;
    float homography_ransac_threshold;
    int homography_max_estimate_iters;
    int homography_max_refine_iters;
    float homography_confidence;
    float homography_outlier_ratio;
    vx_scalar s_fast_threshold;
    vx_scalar s_opt_flow_epsilon;
    vx_scalar s_opt_flow_num_iterations;
    vx_scalar s_opt_flow_use_initial_estimate;
    vx_enum homography_method;
    void init(vx_context context);
    ~ImmediateModeStabilizerParams();
};
class ImmediateModeStabilizer
{
public:
    ImmediateModeStabilizer(vx_context context);
    ~ImmediateModeStabilizer();
    vx_image process(vx_image current_frame);
private:
    virtual void init(vx_image start_frame);
    void findHomogrpahyMatrix(vx_image current_frame);
    void applyPerspectiveTransformation();
    cv::Ptr<HomographySmoother> homography_smoother_;
    std::queue<vx_image> frames;
    ImmediateModeStabilizerParams params_;
    vx_context context_;
    vx_uint32 width_;
    vx_uint32 height_;
    vx_array points_;
    vx_array corresponding_points_;
    vx_image gray_latest_frame_;
    vx_image gray_current_frame_;
    vx_matrix homography_;
    vx_matrix perspective_matrix_;
    vx_image stabilized_frame_;
    vx_pyramid latest_pyr_;
    vx_pyramid current_pyr_;
};
#endif
