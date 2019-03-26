#include "unified.H"

void* UnifiedMemoryClass::operator new(size_t size) {
    void* vp;
    cudaError_t cuda_status = cudaMallocManaged(&vp, size);
    if (cuda_status != cudaSuccess)
        std::cout << cudaGetErrorString(cuda_status) << std::endl;
    assert(cuda_status == cudaSuccess);
    return vp;
}

void UnifiedMemoryClass::operator delete(void* vp) {
    cudaDeviceSynchronize();
    cudaFree(vp);
}
