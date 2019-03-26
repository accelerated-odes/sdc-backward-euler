#include "unified.H"

void* UnifiedMemoryClass::operator new(size_t size) {
    void* vp;
    cudaMallocManaged(&vp, size);
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
        std::cout << cudaGetErrorString(cuda_status) << std::endl;
    assert(cuda_status == cudaSuccess);
    return vp;
}

void UnifiedMemoryClass::operator delete(void* vp) {
    cudaDeviceSynchronize();
    cudaFree(vp);
}
