#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32
#define COARSE_M 4
#define COARSE_N 4
#define BLOCK_M (TILE_SIZE * COARSE_M)
#define BLOCK_N (TILE_SIZE * COARSE_N)

__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x * COARSE_N;
    int row = blockDim.y * blockIdx.y * COARSE_M;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    __shared__ float sA[BLOCK_M][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][BLOCK_N];

    float acc[COARSE_M][COARSE_N] = { 0.0f };
    for (int k = 0; k < K; k += TILE_SIZE) {
        // 타일 복사
        int kx = tidx + k;
        int ky = tidy + k;

        #pragma unroll
        for (int cm = 0; cm < COARSE_M; cm += 1) {
            int ty = TILE_SIZE * cm + tidy; 
            if (kx < K && ty + row < M) {
                sA[ty][tidx] = __half2float(A[(ty + row) * K + kx]);
            } else {
                sA[ty][tidx] = 0.0f;
            }
        }

        #pragma unroll
        for (int cn = 0; cn < COARSE_N; cn += 1) {
            int tx = TILE_SIZE * cn + tidx; 
            if (ky < K && tx + col < N) {
                sB[tidy][tx] = __half2float(B[N * ky + tx + col]);
            } else {
                sB[tidy][tx] = 0.0f;
            }
        }
        __syncthreads();
        
        // 연산
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            // shared -> register 복사
            float rA[COARSE_M];
            float rB[COARSE_N];
            
            #pragma unroll
            for (int cm = 0; cm < COARSE_M; cm++) {
                rA[cm] = sA[tidy + TILE_SIZE * cm][i];
            }
            #pragma unroll
            for (int cn = 0; cn < COARSE_N; cn++) {
                rB[cn] = sB[i][tidx + TILE_SIZE * cn];
            }
            
            #pragma unroll
            for (int cm = 0; cm < COARSE_M; cm++) {
                #pragma unroll
                for (int cn = 0; cn < COARSE_N; cn++) {
                    acc[cm][cn] = fma(rA[cm], rB[cn], acc[cm][cn]);
                }
            }
        }
        __syncthreads();
    }

    // 값 복사
    for (int cm = 0; cm < COARSE_M; cm += 1) {
        for (int cn = 0; cn < COARSE_N; cn += 1) {
            int ty = tidy + TILE_SIZE * cm;
            int tx = tidx + TILE_SIZE * cn;
            int gidx = col + tx;
            int gidy = row + ty;
            if (gidx < N && gidy < M) {
                C[gidy * N + gidx] = __float2half(acc[cm][cn] * alpha + __half2float(C[gidy * N + gidx]) * beta);
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
