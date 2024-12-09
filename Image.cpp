#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

//--------------------------
// 中值滤波函数（手动实现）
// input: 灰度图像
// kernelSize: 滤波器核大小(奇数)
// 无符号字符类型（unsigned char）uchar存储
//--------------------------
cv::Mat medianFilter(const cv::Mat &input, int kernelSize) {
    CV_Assert(input.channels() == 1); // 简化：只处理单通道灰度图像，该断言用于确保输入图像是单通道，即灰度图像。如果输入图像不是单通道，程序将中断，以免在后续处理中出现通道处理的混乱。
    CV_Assert(kernelSize % 2 == 1);   // 只接受核大小必须为奇数

    int Kradius = kernelSize / 2; // 将核大小除以2可获得核半径（Kradius）（框的一半）
    cv::Mat output = input.clone(); //clone保证了输出图像初始值与输入图像相同

    //避免在图像边缘处获取邻域像素时发生越界
    for (int y = Kradius; y < input.rows - Kradius; y++) {
        for (int x = Kradius; x < input.cols - Kradius; x++) {
            //neighbors是一个向量（动态数组），用于存储当前像素 (y, x)邻域内所有像素的灰度值
            std::vector<uchar> neighbors;  
            //预先分配内存，以避免在随后的数据插入中频繁分配空间例如3*3，提高性能。
            neighbors.reserve(kernelSize * kernelSize);

            //这两个内层循环用于遍历 (y, x)像素的邻域区域。
            for (int j = -Kradius; j <= Kradius; j++) {
                for (int i = -Kradius; i <= Kradius; i++) {
                    neighbors.push_back(input.at<uchar>(y + j, x + i));
                }
            }
            //一旦收集了所有邻域像素值，将其排序（从小到大）
            std::sort(neighbors.begin(), neighbors.end()); 
            //排序后的 neighbors 中间位置的元素即为中值。
            uchar medianVal = neighbors[neighbors.size()/2];
            //将中值 medianVal赋予输出图像 output中对应位置 (y, x)的像素，使该像素的值由原来的值变为邻域中值，从而完成中值滤波操作。
            output.at<uchar>(y, x) = medianVal;
        }
    }

    //函数返回中值滤波后的图像 output
    return output;
}

//--------------------------------
// 通用卷积函数（适用于单通道灰度图）
// I * H，其中H为LxL核
// input：输入图像 (I)，类型为 cv::Mat，要求是单通道（灰度图）。
// kernel：卷积核 (H)，也是 cv::Mat 类型，要求是奇数大小（如3×3、5×5）L*L
// 输出：返回卷积后的图像，类型为 cv::Mat。
//--------------------------------
cv::Mat convolve(const cv::Mat &input, const cv::Mat &kernel) {
    CV_Assert(input.channels() == 1);
    // 验证 kernel 是否是方形矩阵，且边长为奇数
    CV_Assert(kernel.rows == kernel.cols && kernel.rows % 2 == 1);

    //确定卷积核的半径，用于确定滑动窗口的范围
    int Kradius = kernel.rows / 2;
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F); // 先用float存储中间结果,初始化为全零矩阵
    
    // 卷积计算
    // 遍历图像中每个有效的像素点
    for (int y = Kradius; y < input.rows - Kradius; y++) {
        for (int x = Kradius; x < input.cols - Kradius; x++) {
            float sum = 0.0f;
            // 遍历以当前像素为中心的邻域区域L*L
            for (int j = -Kradius; j <= Kradius; j++) {
                for (int i = -Kradius; i <= Kradius; i++) {
                    float val = static_cast<float>(input.at<uchar>(y + j, x + i));
                    float k = kernel.at<float>(j + Kradius, i + Kradius);
                    sum += val * k;
                }
            }
            output.at<float>(y, x) = sum;
        }
    }

    // 将结果转换为8位图像
    cv::Mat finalOutput;
    output.convertTo(finalOutput, CV_8U, 1.0, 0.0);

    return finalOutput;
}

//------------------------
// 均值核生成函数，用于卷积
// size: 核大小(奇数)
//------------------------
cv::Mat createMeanKernel(int size) {
    CV_Assert(size % 2 == 1);
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    kernel = kernel / (float)(size*size);
    return kernel;
}

//------------------------
// 高斯核生成函数(简化版本)
// size: 核大小(奇数)
// sigma: 标准差
//------------------------
cv::Mat createGaussianKernel(int size, float sigma) {
    CV_Assert(size % 2 == 1);
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    int radius = size / 2;
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = std::exp(-(x*x + y*y)/(2*sigma*sigma));
            kernel.at<float>(y+radius, x+radius) = val;
            sum += val;
        }
    }
    kernel = kernel / sum;
    return kernel;
}

//------------------------
// 拉普拉斯核 (3x3简单版本)
//------------------------
cv::Mat createLaplacianKernel() {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        0,  1, 0,
        1, -4, 1,
        0,  1, 0);
    return kernel;
}

//------------------------
// Sobel核 (水平Sobel)
//------------------------
cv::Mat createSobelKernelX() {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);
    return kernel;
}

//------------------------
// Sobel核 (垂直Sobel)
//------------------------
cv::Mat createSobelKernelY() {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1);
    return kernel;
}

int main(int argc, char** argv) {
    // 读取输入图像为灰度图（请保证当前目录下有 input.png）
    cv::Mat input = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Error: Impossible de charger l'image ! input.png" << std::endl;
        return -1;
    }

    // 1. 中值滤波
    cv::Mat medianResult = medianFilter(input, 3);

    // 2. 卷积操作示例
    // a) 均值滤波 (3x3)
    cv::Mat meanKernel = createMeanKernel(3);
    cv::Mat meanResult = convolve(input, meanKernel);

    // b) 高斯滤波 (5x5, sigma=1.0)
    cv::Mat gaussKernel = createGaussianKernel(5, 1.0f);
    cv::Mat gaussResult = convolve(input, gaussKernel);

    // 拉普拉斯滤波
    cv::Mat lapKernel = createLaplacianKernel();
    cv::Mat lapResult = convolve(input, lapKernel);

    // Sobel滤波 (水平和垂直方向)
    cv::Mat sobelXKernel = createSobelKernelX();
    cv::Mat sobelYKernel = createSobelKernelY();
    cv::Mat sobelXResult = convolve(input, sobelXKernel);
    cv::Mat sobelYResult = convolve(input, sobelYKernel);

    // 显示结果
    cv::imshow("Original", input);
    cv::imshow("Median Filtered", medianResult);
    cv::imshow("Mean Filtered", meanResult);
    cv::imshow("Gaussian Filtered", gaussResult);
    cv::imshow("Laplacian Filtered", lapResult);
    cv::imshow("Sobel X", sobelXResult);
    cv::imshow("Sobel Y", sobelYResult);

    cv::waitKey(0);

    // 保存结果
    cv::imwrite("median_result.png", medianResult);
    cv::imwrite("mean_result.png", meanResult);
    cv::imwrite("gaussian_result.png", gaussResult);
    cv::imwrite("laplacian_result.png", lapResult);
    cv::imwrite("sobel_x_result.png", sobelXResult);
    cv::imwrite("sobel_y_result.png", sobelYResult);

    return 0;
}

/*
这是一个完整的示例代码（main.cpp），包括：

使用OpenCV导入图像(cv::imread)
显示图像(cv::imshow)
保存处理结果(cv::imwrite)
实现中值滤波函数
实现通用卷积函数以及均值、高斯、拉普拉斯、Sobel核的使用
请根据自己的环境安装OpenCV，并确保编译链接正确。下方给出的编译命令仅为示例，请根据实际路径调整。

说明：
medianFilter函数实现了基本的中值滤波：对每个像素取其邻域内像素的中值作为新值。
convolve函数则是通用的卷积函数。通过传入不同的核（均值、高斯、拉普拉斯、Sobel核）即可实现不同的滤波效果。
createMeanKernel, createGaussianKernel, createLaplacianKernel, createSobelKernelX, createSobelKernelY 函数用于快速生成相应滤波核。
convolve函数中最终输出使用convertTo转为CV_8U。在特定核的场合（如均值或高斯核），理论上卷积结果应天然在0-255范围内（若输入在此范围内），但对差分类核（如拉普拉斯、Sobel）结果可能出现负值，因此这里简单使用convertTo对输出进行一次线性映射。实际应用中可能需要根据需求进行截断或归一化处理。

通过上述代码，即完成了必需的实现：
中值滤波：减少脉冲噪声
卷积及其应用（均值滤波、高斯滤波、拉普拉斯和Sobel边缘检测）
并提供了与OpenCV函数对比的可能（如cv::medianBlur或cv::filter2D）。
*/