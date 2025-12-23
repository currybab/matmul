## T4 GPU

- 1등 기록: 0.7371ms
- [baseline](./01-baseline.cu): 94.83ms (약 129배)
- [tiling](./02-tiling.cu): 4.35ms (약 5.9배)
- [tiling + thread coarsening](./03-tiling_thread_coarsening.cu): 1.45ms (약 1.97배)
- [tensor core](./04-tensor_core.cu): 1.30ms(약 1.76배)
- [tensor core + tiling](./05-tensor_core_tiling.cu)
