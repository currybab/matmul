#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;
    int gidy = threadIdx.y + blockDim.y * blockIdx.y;

    if (gidx >= N || gidy >= M) return;

    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        float rA = __half2float(A[gidy * K + i]);
        float rB = __half2float(B[i * N + gidx]);

        acc = fma(rA, rB, acc);
    }
    C[gidy * N + gidx] = __float2half(acc * alpha + __half2float(C[gidy * N + gidx]) * beta);
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(N, M);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
