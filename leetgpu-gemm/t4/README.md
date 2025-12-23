## T4 GPU

- 기존 1등 기록: 0.7371ms
- [baseline](./01-baseline.cu): 94.83ms (약 129배)
- [tiling](./02-tiling.cu): 4.35ms (약 5.9배)
- [tiling + thread coarsening](./03-tiling_thread_coarsening.cu): 1.45ms (약 1.97배)
- [WMMA](./04-wmma.cu): 1.30ms(약 1.76배)
    - tensor core 사용하는 부분에서 warp 단위의 반강제 타일링(?)이 적용된 버전이 cuda core만 사용하는 tiling + thread coarsening 버전보다 10% 정도 더 빨랐음. 
- [WMMA + tiling](./05-wmma_tiling.cu): 0.7174ms (2.67% 개선)
- WMMA + tiling + double buffer: not implemented yet
- PTX 적용: not implemented yet

## T4 WMMA API

1. Initializing `wmma::fragments`.
2. Loading data into fragments using functions like `wmma::load_matrix_sync`.
3. Performing the matrix multiplication and accumulation using `wmma::mma_sync`.
4. Storing the results back to memory using `wmma::store_matrix_sync`.

```cpp
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);

// ldm = 메모리에서 한 행의 시작부터 다음 행의 시작까지의 원소 개수
// layout_t = 행렬이 메모리에 저장된 방식 
// wmma::row_major  // 행 우선 (C 스타일)
// wmma::col_major  // 열 우선 (Fortran 스타일)
// wmma::mem_row_major  // store용 row-major
// wmma::mem_col_major  // store용 col-major
```
