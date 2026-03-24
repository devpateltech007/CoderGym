#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct SizeConfig {
    int K;
    int N;
};

static void checkCuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error (%s): %d\n", msg, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

static std::vector<SizeConfig> parseSizes(const std::string& sizes) {
    std::vector<SizeConfig> out;
    std::stringstream ss(sizes);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto x_pos = token.find('x');
        if (x_pos == std::string::npos) {
            std::fprintf(stderr, "Invalid size token: %s\n", token.c_str());
            std::exit(EXIT_FAILURE);
        }
        int left = std::stoi(token.substr(0, x_pos));
        int right = std::stoi(token.substr(x_pos + 1));
        out.push_back(SizeConfig{left, right});
    }
    return out;
}

static void fillHost(float* ptr, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = 0.001f * static_cast<float>(i % 127);
    }
}

static float benchmarkSgemm(cublasHandle_t handle, int M, int N, int K, int iters, int warmup) {
    float *A, *B, *C;
    size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeB = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(N);

    checkCuda(cudaMalloc(&A, sizeA * sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc(&B, sizeB * sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc(&C, sizeC * sizeof(float)), "cudaMalloc C");

    float* hA = static_cast<float*>(std::malloc(sizeA * sizeof(float)));
    float* hB = static_cast<float*>(std::malloc(sizeB * sizeof(float)));
    fillHost(hA, sizeA);
    fillHost(hB, sizeB);
    checkCuda(cudaMemcpy(A, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(B, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    std::free(hA);
    std::free(hB);

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        checkCublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N),
            "cublasSgemm warmup");
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");
    checkCuda(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iters; ++i) {
        checkCublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N),
            "cublasSgemm");
    }
    checkCuda(cudaEventRecord(stop), "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    float avg_ms = elapsed_ms / static_cast<float>(iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return avg_ms;
}

static float benchmarkTf32(cublasHandle_t handle, int M, int N, int K, int iters, int warmup) {
    float *A, *B, *C;
    size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeB = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(N);

    checkCuda(cudaMalloc(&A, sizeA * sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc(&B, sizeB * sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc(&C, sizeC * sizeof(float)), "cudaMalloc C");

    float* hA = static_cast<float*>(std::malloc(sizeA * sizeof(float)));
    float* hB = static_cast<float*>(std::malloc(sizeB * sizeof(float)));
    fillHost(hA, sizeA);
    fillHost(hB, sizeB);
    checkCuda(cudaMemcpy(A, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(B, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    std::free(hA);
    std::free(hB);

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        checkCublas(
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K,
                         &beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx warmup");
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");
    checkCuda(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iters; ++i) {
        checkCublas(
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K,
                         &beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    }
    checkCuda(cudaEventRecord(stop), "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    float avg_ms = elapsed_ms / static_cast<float>(iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return avg_ms;
}

int main(int argc, char** argv) {
    int batch_size = 1024;
    int warmup = 10;
    int iters = 50;
    std::string sizes_arg = "1024x1024,2048x2048,4096x4096,8192x8192";
    std::string out_csv = "Excellent_option/results/cublas_gemm_results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--sizes" && i + 1 < argc) {
            sizes_arg = argv[++i];
        } else if (arg == "--out-csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete arg: %s\n", arg.c_str());
            return EXIT_FAILURE;
        }
    }

    auto sizes = parseSizes(sizes_arg);
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    std::ofstream out(out_csv);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open output CSV: %s\n", out_csv.c_str());
        return EXIT_FAILURE;
    }
    out << "framework,mode,batch_size,in_features,out_features,avg_ms,tflops,speedup_vs_sgemm\n";

    struct Row {
        int k;
        int n;
        float avg_ms;
        double tflops;
    };
    std::vector<Row> baseline;

    for (const auto& s : sizes) {
        int M = batch_size;
        int K = s.K;
        int N = s.N;
        float ms = benchmarkSgemm(handle, M, N, K, iters, warmup);
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        baseline.push_back(Row{K, N, ms, tflops});
        out << "cublas,sgemm," << M << "," << K << "," << N << "," << ms << "," << tflops << ",1.0\n";
        std::printf("[cublas][sgemm] M=%d K=%d N=%d avg_ms=%.6f tflops=%.4f\n", M, K, N, ms, tflops);
    }

    checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "set TF32 math mode");
    for (size_t i = 0; i < sizes.size(); ++i) {
        int M = batch_size;
        int K = sizes[i].K;
        int N = sizes[i].N;
        float ms = benchmarkTf32(handle, M, N, K, iters, warmup);
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double tflops = flops / (ms * 1e-3) / 1e12;
        double speedup = baseline[i].avg_ms / ms;
        out << "cublas,tf32_tensor_op," << M << "," << K << "," << N << "," << ms << "," << tflops << "," << speedup
            << "\n";
        std::printf("[cublas][tf32]  M=%d K=%d N=%d avg_ms=%.6f tflops=%.4f speedup=%.3fx\n", M, K, N, ms, tflops,
                    speedup);
    }

    out.close();
    cublasDestroy(handle);
    std::printf("Saved CSV to: %s\n", out_csv.c_str());
    return EXIT_SUCCESS;
}
