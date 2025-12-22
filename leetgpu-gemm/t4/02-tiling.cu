#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;
    int gidy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE) {
        // 타일 복사
        int tile_x = threadIdx.x + tile_idx;
        int tile_y = threadIdx.y + tile_idx;
        if (tile_x < K && gidy < M) {
            sA[threadIdx.y][threadIdx.x] = __half2float(A[gidy * K + tile_x]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile_y < K && gidx < N) {
            sB[threadIdx.y][threadIdx.x] = __half2float(B[N * tile_y + gidx]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        // 연산
        for (int i = 0; i < TILE_SIZE; i++) {
            acc = fma(sA[threadIdx.y][i], sB[i][threadIdx.x], acc);
        }
        __syncthreads();
    }

    // 값 복사
    if (gidx < N && gidy < M) {
        C[gidy * N + gidx] = __float2half(acc * alpha + __half2float(C[gidy * N + gidx]) * beta);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
