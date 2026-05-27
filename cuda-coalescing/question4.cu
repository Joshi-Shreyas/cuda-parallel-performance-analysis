#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

inline void cudaCheck(cudaError_t err, const char* fl, int li) { if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d - %s\n", fl, li, cudaGetErrorString(err)); exit(EXIT_FAILURE); } }
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__global__ void matvec_coalesced(const float* __restrict__ A, const float* __restrict__ x, float*  __restrict__ y, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (int col = 0; col < N; col++)
        sum += A[col * M + row] * x[col];
    y[row] = sum;
}

__global__ void matvec_noncoalesced(const float* __restrict__ A, const float* __restrict__ x, float*  __restrict__ y, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.0f;
    for (int col = 0; col < N; col++)
        sum += A[row * N + col] * x[col];
    y[row] = sum;
}

static void rand_fill(float* a, size_t n) { for (size_t i = 0; i < n; i++) a[i] = (float)rand() / RAND_MAX; }

static void to_col_major(const float* Ar, float* Ac, int M, int N) {
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            Ac[c * M + r] = Ar[r * N + c];
}

static float time_kernel(void (*launch)(const float*, const float*, float*, int, int, dim3, dim3), const float* dA, const float* dx, float* dy, int M, int N, dim3 grid, dim3 block, int reps)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch(dA, dx, dy, M, N, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < reps; i++) launch(dA, dx, dy, M, N, grid, block);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms / reps;
}

static void launch_c (const float* dA, const float* dx, float* dy, int M, int N, dim3 g, dim3 b) { matvec_coalesced   <<<g,b>>>(dA,dx,dy,M,N); }
static void launch_nc(const float* dA, const float* dx, float* dy, int M, int N, dim3 g, dim3 b) { matvec_noncoalesced<<<g,b>>>(dA,dx,dy,M,N); }

static void benchmark(int M, int N, int reps)
{
    size_t matB = (size_t)M*N*sizeof(float), vecB = N*sizeof(float), outB = M*sizeof(float);

    float *hAr = (float*)malloc(matB), *hAc = (float*)malloc(matB);
    float *hx  = (float*)malloc(vecB), *hy_c = (float*)malloc(outB), *hy_nc = (float*)malloc(outB);
    rand_fill(hAr, (size_t)M*N); rand_fill(hx, N);
    to_col_major(hAr, hAc, M, N);

    float *dAr, *dAc, *dx, *dy_c, *dy_nc;
    CUDA_CHECK(cudaMalloc(&dAr,  matB)); CUDA_CHECK(cudaMalloc(&dAc,  matB));
    CUDA_CHECK(cudaMalloc(&dx,   vecB)); CUDA_CHECK(cudaMalloc(&dy_c, outB)); CUDA_CHECK(cudaMalloc(&dy_nc, outB));
    CUDA_CHECK(cudaMemcpy(dAr, hAr, matB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAc, hAc, matB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx,  hx,  vecB, cudaMemcpyHostToDevice));

    dim3 block(256), grid((M+255)/256);

    float t_c  = time_kernel(launch_c,  dAc, dx, dy_c,  M, N, grid, block, reps);
    float t_nc = time_kernel(launch_nc, dAr, dx, dy_nc, M, N, grid, block, reps);

    double bytes = (double)matB + vecB + outB;
    double bw_c  = bytes / (t_c  * 1e-3) / 1e9;
    double bw_nc = bytes / (t_nc * 1e-3) / 1e9;

    CUDA_CHECK(cudaMemcpy(hy_c,  dy_c,  outB, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy_nc, dy_nc, outB, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < M; i++) { float d = fabsf(hy_c[i]-hy_nc[i]); if (d>max_err) max_err=d; }

    printf("  M=%-6d N=%-6d | Coalesced: %7.3f ms  %6.2f GB/s | Non-coalesced: %7.3f ms  %6.2f GB/s | Speedup: %.2fx | MaxErr: %.2e\n", M, N, t_c, bw_c, t_nc, bw_nc, t_nc/t_c, max_err);

    free(hAr); free(hAc); free(hx); free(hy_c); free(hy_nc);
    CUDA_CHECK(cudaFree(dAr)); CUDA_CHECK(cudaFree(dAc)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy_c)); CUDA_CHECK(cudaFree(dy_nc));
}

int main(void)
{
    srand(42);
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s | SM %d.%d | Peak BW: %.1f GB/s\n\n", p.name, p.major, p.minor, 2.0*p.memoryClockRate*(p.memoryBusWidth/8)/1e6);

    int sizes[] = {512, 1024, 2048, 4096, 8192, 16384};
    for (int i = 0; i < 6; i++) benchmark(sizes[i], sizes[i], 20);

    printf("\n(Each timing is an average over 20 kernel launches.)\n");
    return 0;
}
