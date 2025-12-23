#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define WARP_SIZE 32
using namespace nvcuda;

// WMMA T4(Turing) - m16n16k16
// D[16x16] = A[16x16] * B[16x16] + C[16x16]
constexpr int WMMA_M = 16;  
constexpr int WMMA_N = 16;  // B, C의 열 수  
constexpr int WMMA_K = 16;  // A의 열 수 = B의 행 수

// 블록당 4x4 = 16개의 warp 배치
// 각 warp은 하나의 16x16 WMMA 타일을 담당
constexpr int WARP_M = 4;   // M 방향 warp 수
constexpr int WARP_N = 4;   // N 방향 warp 수

// 블록이 처리하는 총 영역 크기
constexpr int BLOCK_M = WMMA_M * WARP_M;
constexpr int BLOCK_N = WMMA_N * WARP_N;
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
    const int warpRowOffset = warpRow * WMMA_M;
    const int warpColOffset = warpCol * WMMA_N;

    __shared__ half sA[BLOCK_M][BLOCK_K];
    __shared__ half sB[BLOCK_K][BLOCK_N];
    __shared__ half sC[BLOCK_M][BLOCK_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; // accumulator만 float
    wmma::fill_fragment(acc, 0.0f);  // 0으로 초기화

    
    // main loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // load tiles
        #pragma unroll
        for (int i = 0; i < WARP_M; i++) {
            // 선형 인덱스 계산: 각 스레드가 담당할 원소
            int idx = threadIdx.x + i * NUM_THREADS;
            
            // 2D 인덱스로 변환 (row-major 순서로 순회)
            int row = idx / BLOCK_K;  // 0~63
            int col = idx % BLOCK_K;  // 0~31
            
            // 전역 메모리 좌표
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            
            // 경계 체크 후 로드 (범위 밖이면 0)
            if (globalRow < M && globalCol < K) {
                sA[row][col] = A[globalRow * K + globalCol];
            } else {
                sA[row][col] = __float2half(0.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < WARP_N; i++) {
            int idx = threadIdx.x + i * NUM_THREADS;
            
            int row = idx / BLOCK_N;  // 0~31
            int col = idx % BLOCK_N;  // 0~63
            
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
        for (int wk = 0; wk < BLOCK_K; wk += WMMA_K) {
            wmma::load_matrix_sync(fragA, &sA[warpRowOffset][wk], BLOCK_K);
            wmma::load_matrix_sync(fragB, &sB[wk][warpColOffset], BLOCK_N);
            
            wmma::mma_sync(acc, fragA, fragB, acc);
        }
        __syncthreads();
    }

    if (beta != 0.0f) {
        #pragma unroll
        for (int i = 0; i < (BLOCK_M * BLOCK_N) / NUM_THREADS; i++) {
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

        // Shared → Register: 기존 C fragment 로드
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragC;
        wmma::load_matrix_sync(fragC, &sC[warpRowOffset][warpColOffset], BLOCK_N, wmma::mem_row_major);
        
        // alpha * acc + beta * C 계산
        // fragment 내 각 원소에 대해 수행
        #pragma unroll
        for (int i = 0; i < acc.num_elements; i++) {
            acc.x[i] = alpha * acc.x[i] + beta * __half2float(fragC.x[i]);
        }
    } else {
        // beta == 0: alpha 스케일링만 적용
        #pragma unroll
        for (int i = 0; i < acc.num_elements; i++) {
            acc.x[i] *= alpha;
        }
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragOut;
    #pragma unroll
    for (int i = 0; i < acc.num_elements; i++) {
        fragOut.x[i] = __float2half(acc.x[i]);
    }
    
    wmma::store_matrix_sync(&sC[warpRowOffset][warpColOffset], fragOut, BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < (BLOCK_M * BLOCK_N) / NUM_THREADS; i++) {
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
