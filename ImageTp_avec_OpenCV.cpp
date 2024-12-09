#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    // 请确保argv[1]为输入图像的路径
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    // 1. 读取原始图像 (彩色或者灰度)
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Impossible de charger l'image !" << std::endl;
        return -1;
    }

    // 2. 转灰度图像以便于处理（如果需要）
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    // 显示原图
    cv::imshow("Original", gray);

    // -----------------------------
    // 3. 中值滤波
    // -----------------------------
    cv::Mat medianFiltered;
    int kernelSizeMedian = 3; // 中值滤波核大小（奇数）
    cv::medianBlur(gray, medianFiltered, kernelSizeMedian);
    cv::imshow("Median Filtered", medianFiltered);

    // -----------------------------
    // 4. 卷积操作
    // -----------------------------
    // (a) 均值滤波（通过卷积）
    // 构造一个简单的平均核(如3x3)
    int kernelSizeMean = 3;
    cv::Mat meanKernel = cv::Mat::ones(kernelSizeMean, kernelSizeMean, CV_32F) / (float)(kernelSizeMean * kernelSizeMean);

    cv::Mat meanFiltered;
    cv::filter2D(gray, meanFiltered, -1, meanKernel);
    cv::imshow("Mean Filtered", meanFiltered);

    // (b) 高斯滤波(使用高斯核进行卷积)
    // 在此使用OpenCV自带的getGaussianKernel来创建一维高斯核，然后组合成2D核
    int kernelSizeGauss = 5;
    double sigma = 1.0; // 高斯分布标准差，可调
    cv::Mat gaussKernelX = cv::getGaussianKernel(kernelSizeGauss, sigma, CV_32F);
    cv::Mat gaussKernelY = cv::getGaussianKernel(kernelSizeGauss, sigma, CV_32F);
    cv::Mat gaussKernel2D = gaussKernelX * gaussKernelY.t(); // 外积生成2D高斯核

    cv::Mat gaussFiltered;
    cv::filter2D(gray, gaussFiltered, -1, gaussKernel2D);
    cv::imshow("Gaussian Filtered", gaussFiltered);

    // (c) 使用差分滤波器（如Sobel算子）检测边缘、梯度
    // Sobel算子内置函数直接使用：
    cv::Mat sobelX, sobelY, sobelMag;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0); // 对x方向求梯度
    cv::Sobel(gray, sobelY, CV_32F, 0, 1); // 对y方向求梯度

    // 计算梯度幅度
    cv::magnitude(sobelX, sobelY, sobelMag);
    // 归一化以便显示
    double minVal, maxVal;
    cv::minMaxLoc(sobelMag, &minVal, &maxVal);
    cv::Mat sobelDisplay;
    sobelMag.convertTo(sobelDisplay, CV_8U, 255.0 / maxVal);

    cv::imshow("Sobel Magnitude", sobelDisplay);

    // 如果想对Sobel核进行手工卷积：可自行构建Sobel核后用filter2D卷积
    // Sobel x方向核（3x3）
    float sobelXData[9] = {-1, 0, 1,
                           -2, 0, 2,
                           -1, 0, 1};
    cv::Mat sobelKernelX(3, 3, CV_32F, sobelXData);

    cv::Mat customSobelX;
    cv::filter2D(gray, customSobelX, CV_32F, sobelKernelX);
    // 同理可实现sobelY核，然后计算幅度

    // 显示手动卷积得到的SobelX结果（归一化）
    cv::Mat customSobelXDisplay;
    cv::normalize(customSobelX, customSobelXDisplay, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Custom Sobel X", customSobelXDisplay);

    // -----------------------------
    // 5. 比较结果
    // -----------------------------
    // 您可以肉眼观察结果，或使用PSNR/MSE等指标进行客观比较。
    // 对BONUS（加分项）：可将您自己实现的卷积函数结果与cv::filter2D结果进行对比，验证正确性。

    cv::waitKey(0);
    return 0;
}
