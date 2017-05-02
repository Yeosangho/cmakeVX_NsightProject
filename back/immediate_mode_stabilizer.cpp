#include "immediate_mode_stabilizer.hpp"
#include "NVX/nvx_opencv_interop.hpp"
#include "NVX/nvx_timer.hpp"
#include "OVX/UtilityOVX.hpp"
void ImmediateModeStabilizerParams::init(vx_context context)
{
    smoothing_window_size = 8;
    fast_threshold = 25.f;
    num_pyramid_levels = 3;
    opt_flow_win_size = 21;
    opt_flow_epsilon = 0.01f;
    opt_flow_num_iterations = 30;
    opt_flow_use_initial_estimate = 0;
    homography_ransac_threshold = 3.f;
    homography_max_estimate_iters = 2000;
    homography_max_refine_iters = 10;
    homography_confidence = 0.995f;
    homography_outlier_ratio = 0.45f;
    s_fast_threshold = vxCreateScalar(context, VX_TYPE_FLOAT32, &fast_threshold);
    NVXIO_CHECK_REFERENCE(s_fast_threshold);
    homography_method = NVX_FIND_HOMOGRAPHY_METHOD_RANSAC;
    s_opt_flow_epsilon = vxCreateScalar(context, VX_TYPE_FLOAT32, &opt_flow_epsilon);
    NVXIO_CHECK_REFERENCE(s_opt_flow_epsilon);
    s_opt_flow_num_iterations = vxCreateScalar(context, VX_TYPE_UINT32, &opt_flow_num_iterations);
    NVXIO_CHECK_REFERENCE(s_opt_flow_num_iterations);
    assert(opt_flow_use_initial_estimate == 0);
    vx_bool use_initial_estimate = vx_false_e;
    s_opt_flow_use_initial_estimate = vxCreateScalar(context, VX_TYPE_BOOL, &use_initial_estimate);
    NVXIO_CHECK_REFERENCE(s_opt_flow_use_initial_estimate);
}
ImmediateModeStabilizer::ImmediateModeStabilizer(vx_context context)
{
    context_ = context;
    homography_smoother_ = 0;
    width_ = 0;
    height_ = 0;
    points_ = 0;
    corresponding_points_ = 0;
    gray_latest_frame_ = 0;
    gray_current_frame_ = 0;
    homography_ = 0;
    perspective_matrix_ = 0;
    stabilized_frame_ = 0;
    latest_pyr_ = 0;
    current_pyr_ = 0;
}
ImmediateModeStabilizerParams::~ImmediateModeStabilizerParams()
{
    vxReleaseScalar(&s_fast_threshold);
    vxReleaseScalar(&s_opt_flow_epsilon);
    vxReleaseScalar(&s_opt_flow_num_iterations);
    vxReleaseScalar(&s_opt_flow_use_initial_estimate);
}
ImmediateModeStabilizer::~ImmediateModeStabilizer()
{
    while(!frames.empty())
    {
        vx_image image = frames.front();
        vxReleaseImage(&image);
        frames.pop();
    }
    vxReleaseArray(&points_);
    vxReleaseArray(&corresponding_points_);
    vxReleaseImage(&gray_latest_frame_);
    vxReleaseImage(&gray_current_frame_);
    vxReleaseMatrix(&homography_);
    vxReleaseMatrix(&perspective_matrix_);
    vxReleaseImage(&stabilized_frame_);
    vxReleasePyramid(&latest_pyr_);
    vxReleasePyramid(&current_pyr_);
};
void ImmediateModeStabilizer::init(vx_image start_frame)
{
    params_.init(context_);
    homography_smoother_ = new HomographySmoother(params_.smoothing_window_size);
    const int array_type = NVX_TYPE_KEYPOINTF;
    const vx_uint32 array_capacity = 15000;
    points_ = vxCreateArray(context_, array_type, array_capacity);
    NVXIO_CHECK_REFERENCE(points_);
    corresponding_points_ = vxCreateArray(context_, array_type, array_capacity);
    NVXIO_CHECK_REFERENCE(corresponding_points_);
    NVXIO_SAFE_CALL(vxQueryImage(start_frame, VX_IMAGE_ATTRIBUTE_WIDTH, &width_, sizeof(width_)));
    NVXIO_SAFE_CALL(vxQueryImage(start_frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_, sizeof(height_)));
    gray_latest_frame_ = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(gray_latest_frame_);
    NVXIO_SAFE_CALL(vxuColorConvert(context_, start_frame, gray_latest_frame_));
    gray_current_frame_ = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(gray_current_frame_);
    latest_pyr_ = vxCreatePyramid(context_,
                                  (vx_size)params_.num_pyramid_levels,
                                  VX_SCALE_PYRAMID_HALF,
                                  width_,
                                  height_,
                                  VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(latest_pyr_);
    NVXIO_SAFE_CALL(vxuGaussianPyramid(context_, gray_latest_frame_, latest_pyr_));
    current_pyr_ = vxCreatePyramid(context_,
                                   (vx_size)params_.num_pyramid_levels,
                                   VX_SCALE_PYRAMID_HALF,
                                   width_,
                                   height_,
                                   VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(current_pyr_);
    homography_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
    NVXIO_CHECK_REFERENCE(homography_);
    perspective_matrix_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
    NVXIO_CHECK_REFERENCE(perspective_matrix_);
    stabilized_frame_ = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(stabilized_frame_);
};
void ImmediateModeStabilizer::findHomogrpahyMatrix(vx_image current_frame)
{
    NVX_TIMER(fast_corners, "FastCorners");
    NVXIO_SAFE_CALL(vxuFastCorners(context_, gray_latest_frame_, params_.s_fast_threshold, vx_true_e, points_, 0));
    NVX_TIMEROFF(fast_corners);
    NVX_TIMER(color_convert, "ColorConvert");
    NVXIO_SAFE_CALL(vxuColorConvert(context_, current_frame, gray_current_frame_));
    NVX_TIMEROFF(color_convert);
    NVX_TIMER(build_pyramid, "BuildPyramid");
    NVXIO_SAFE_CALL(vxuGaussianPyramid(context_, gray_current_frame_, current_pyr_));
    NVX_TIMEROFF(build_pyramid);
    NVX_TIMER(optical_flow, "OpticalFlow");
    NVXIO_SAFE_CALL(vxuOpticalFlowPyrLK(context_,
                                        latest_pyr_,
                                        current_pyr_,
                                        points_,
                                        points_,
                                        corresponding_points_,
                                        VX_TERM_CRITERIA_BOTH,
                                        params_.s_opt_flow_epsilon,
                                        params_.s_opt_flow_num_iterations,
                                        params_.s_opt_flow_use_initial_estimate,
                                        params_.opt_flow_win_size
                        )
        );
    NVX_TIMEROFF(optical_flow);
    std::swap(latest_pyr_, current_pyr_);
    std::swap(gray_latest_frame_, gray_current_frame_);
    NVX_TIMER(find_homography, "FindHomography");
    NVXIO_SAFE_CALL(nvxuFindHomography(context_,
                                       points_,
                                       corresponding_points_,
                                       homography_,
                                       params_.homography_method,
                                       params_.homography_ransac_threshold,
                                       params_.homography_max_estimate_iters,
                                       params_.homography_max_refine_iters,
                                       params_.homography_confidence,
                                       params_.homography_outlier_ratio,
                                       NULL
                        )
        );
    NVX_TIMEROFF(find_homography);
};
void ImmediateModeStabilizer::applyPerspectiveTransformation()
{
    vx_image oldest_frame = frames.front();
    NVX_TIMER(warp_perspective, "WarpPerspective");
    NVXIO_SAFE_CALL(vxuWarpPerspective(context_,
                                       oldest_frame,
                                       perspective_matrix_,
                                       VX_INTERPOLATION_TYPE_BILINEAR,
                                       stabilized_frame_
                        )
        );
    NVX_TIMEROFF(warp_perspective);
    vxReleaseImage(&oldest_frame);
    frames.pop();
};
vx_image ImmediateModeStabilizer::process(vx_image current_frame)
{
    // Initialization
    if (frames.empty())
    {
        init(current_frame);
        vx_image frame = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
        NVXIO_CHECK_REFERENCE(frame);
        NVXIO_SAFE_CALL(nvxuCopyImage(context_, current_frame, frame));
        frames.push(frame);
        return vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
    }
    vx_image frame = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
    NVXIO_SAFE_CALL(nvxuCopyImage(context_, current_frame, frame));
    frames.push(frame);
    findHomogrpahyMatrix(frame);
    cv::Mat homography;
    nvx_cv::copyVXMatrixToCVMat(homography_, homography);
    homography_smoother_->push(homography);
    cv::Matx33f transformation;
    if (!homography_smoother_->getSmoothedHomography(transformation))
    {
        return vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
    }
    transformation = transformation.inv();
    nvx_cv::copyCVMatToVXMatrix(cv::Mat(transformation), perspective_matrix_);
    applyPerspectiveTransformation();
    return stabilized_frame_;
};
