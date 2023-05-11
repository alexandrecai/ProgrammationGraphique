#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void boxBlur(const cv::cuda::PtrStep<uchar3> src,
                        cv::cuda::PtrStep<uchar3> dst,
                        const int kernel_size) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < src.cols && y < src.rows) {
        float3 sum = make_float3(0.f, 0.f, 0.f);
        int count = 0;
        for (int dy = -kernel_size; dy <= kernel_size; ++dy) {
            for (int dx = -kernel_size; dx <= kernel_size; ++dx) {
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx >= 0 && nx < src.cols && ny >= 0 && ny < src.rows) {
                    const uchar3 p = src(ny, nx);
                    sum += make_float3(p.x, p.y, p.z);
                    count += 1;
                }
            }
        }
        const float scale = 1.f / static_cast<float>(count);
        const uchar3 avg = make_uchar3(sum.x * scale, sum.y * scale, sum.z * scale);
        dst(y, x) = avg;
    }
}

int main() {
    cv::Mat img = cv::imread("../images/input.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_src(img);
    cv::cuda::GpuMat d_dst(d_src.size(), d_src.type());

    const dim3 block_size(32, 32);
    const dim3 grid_size((d_src.cols - 1) / block_size.x + 1,
                         (d_src.rows - 1) / block_size.y + 1);
    const int kernel_size = 3;

    // Streams declaration.
    cudaStream_t streams[2];

    // Creation.
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Copying input image to device by halves.
    cudaMemcpyAsync(d_src.ptr(), img.data, d_src.cols * d_src.rows * sizeof(uchar3) / 2,
                    cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_src.ptr(d_src.rows / 2), img.data + d_src.cols * d_src.rows * sizeof(uchar3) / 2,
                    d_src.cols * d_src.rows * sizeof(uchar3) / 2, cudaMemcpyHostToDevice, streams[1]);

    // Launching one kernel in each stream.
    boxBlur<<<grid_size, block_size, 0, streams[0]>>>(d_src, d_dst, kernel_size);
    boxBlur<<<grid_size, block_size, 0, streams[1]>>>(d_src.ptr(d_src.rows / 2), d_dst.ptr(d_src.rows / 2), kernel_size);

    // Copying output image from device by halves.
    cudaMemcpyAsync(img.data, d_dst.ptr(), d_src.cols * d_src.rows * sizeof(uchar3) / 2,
                    cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(img.data + d_src.cols * d_src.rows * sizeof(uchar3) / 2, d_dst.ptr(d_src.rows / 2),
                    d_src.cols * d_src.rows *
                            sizeof(uchar3) / 2, cudaMemcpyDeviceToHost, streams[1]);

// Synchronizing streams.
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

// Destroying streams.
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    cv::imwrite("out.jpg", img);

    return 0;
}