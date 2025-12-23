#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int WARP_SIZE = 32;

// WMMA T4(Turing) - m16n16k16
// D[16x16] = A[16x16] * B[16x16] + C[16x16]
constexpr int WMMA_M = 16;  
constexpr int WMMA_N = 16;  // B, C의 열 수  
constexpr int WMMA_K = 16;  // A의 열 수 = B의 행 수

constexpr int TILE_M = 2;  // M 방향 타일 수 (warp)
constexpr int TILE_N = 2;  // N 방향 타일 수 (warp)

// 블록당 4x4 = 16개의 warp 배치
// 각 warp은 하나의 16x16 WMMA 타일을 담당
constexpr int WARP_M = 4;   // M 방향 warp 수
constexpr int WARP_N = 4;   // N 방향 warp 수

// 블록이 처리하는 총 영역 크기
constexpr int BLOCK_M = WMMA_M * TILE_M * WARP_M;
constexpr int BLOCK_N = WMMA_N * TILE_N * WARP_N;
constexpr int BLOCK_K = 32;

constexpr int NUM_THREADS = WARP_M * WARP_N * WARP_SIZE;

__global__ void matmul_wmma(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    
    const int warpId = threadIdx.x / WARP_SIZE; // 스레드가 속한 warp id
    const int warpRow = warpId / WARP_N;  // 어느 행의 warp인지
    const int warpCol = warpId % WARP_N;  // 어느 열의 warp인지

    // 블록이 담당하는 C 행렬의 시작 좌표
    const int blockRowStart = blockIdx.y * BLOCK_M;  // M 방향
    const int blockColStart = blockIdx.x * BLOCK_N;  // N 방향

    // warp가 담당하는 16x16 타일의 블록 내 시작 위치
    const int warpRowStart = warpRow * WMMA_M * TILE_M;
    const int warpColStart = warpCol * WMMA_N * TILE_N;

    __shared__ half sA[BLOCK_M][BLOCK_K];
    __shared__ half sB[BLOCK_K][BLOCK_N];
    __shared__ half sC[BLOCK_M][BLOCK_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA[TILE_M]; // (bM) * bK
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragB[TILE_N]; // bK * (bN)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[TILE_M][TILE_N]; // accumulator만 float
    
    #pragma unroll
    for (int tm = 0; tm < TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            wmma::fill_fragment(acc[tm][tn], 0.0f);
        }
    }

    
    // main loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // load tiles
        #pragma unroll
        for (int i = 0; i < (BLOCK_M * BLOCK_K) / NUM_THREADS ; i++) {
            int idx = threadIdx.x + i * NUM_THREADS;
            
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
        
            if (globalRow < M && globalCol < K) {
                sA[row][col] = A[globalRow * K + globalCol];
            } else {
                sA[row][col] = __float2half(0.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < (BLOCK_N * BLOCK_K) / NUM_THREADS; i++) {
            int idx = threadIdx.x + i * NUM_THREADS;
            
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            
            int globalRow = k + row;
            int globalCol = blockColStart + col;
            
            if (globalRow < K && globalCol < N) {
                sB[row][col] = B[globalRow * N + globalCol];
            } else {
                sB[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();

       
        // WMMA 연산 
        #pragma unroll
        for (int bk = 0; bk < BLOCK_K; bk += WMMA_K) {
            #pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                int aRow = warpRowStart + tm * WMMA_M;
                wmma::load_matrix_sync(fragA[tm], &sA[aRow][bk], BLOCK_K);
            }
            
            #pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int bCol = warpColStart + tn * WMMA_N;
                wmma::load_matrix_sync(fragB[tn], &sB[bk][bCol], BLOCK_N);
            }
            
            #pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    wmma::mma_sync(acc[tm][tn], fragA[tm], fragB[tn], acc[tm][tn]);
                }
            }
        }
        __syncthreads();
    }


    // epilogue
    constexpr int LOAD_C = (BLOCK_M * BLOCK_N) / NUM_THREADS;
    if (beta != 0.0f) {
        #pragma unroll
        for (int i = 0; i < LOAD_C; i++) {
            int idx = threadIdx.x + i * NUM_THREADS;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int globalRow = blockRowStart + row;
            int globalCol = blockColStart + col;
            
            if (globalRow < M && globalCol < N) {
                sC[row][col] = C[globalRow * N + globalCol];
            } else {
                sC[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tm = 0; tm < TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragC;
                wmma::load_matrix_sync(fragC, &sC[warpRowStart + tm * WMMA_M][warpColStart + tn * WMMA_N], 
                                       BLOCK_N, wmma::mem_row_major);
                #pragma unroll
                for (int i = 0; i < acc[tm][tn].num_elements; i++) {
                    acc[tm][tn].x[i] = alpha * acc[tm][tn].x[i] + beta * __half2float(fragC.x[i]);
                }
            }
        }
    } else {
        #pragma unroll
        for (int tm = 0; tm < TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                #pragma unroll
                for (int i = 0; i < acc[tm][tn].num_elements; i++) {
                    acc[tm][tn].x[i] *= alpha;
                }
            }
        }
    }

    // register to smem
    #pragma unroll
    for (int tm = 0; tm < TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragOut;
            #pragma unroll
            for (int i = 0; i < acc[tm][tn].num_elements; i++) {
                fragOut.x[i] = __float2half(acc[tm][tn].x[i]);
            }
            wmma::store_matrix_sync(&sC[warpRowStart + tm * WMMA_M][warpColStart + tn * WMMA_N], fragOut, BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();


    // save result
    #pragma unroll
    for (int i = 0; i < LOAD_C; i++) {
        int idx = threadIdx.x + i * NUM_THREADS;
        int row = idx / BLOCK_N;
        int col = idx % BLOCK_N;
        int globalRow = blockRowStart + row;
        int globalCol = blockColStart + col;
        
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = sC[row][col];
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    matmul_wmma<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
