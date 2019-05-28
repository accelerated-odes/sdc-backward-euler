#ifndef _VECTOR_STORAGE_H
#define _VECTOR_STORAGE_H
#include "VectorGpuMacros.H"
#include "AMReX_Array.H"

#ifndef AMREX_USE_CUDA
using std::min;
using std::max;
#endif

template<class StoreType, size_t alloc_size>
class StackMemory {
public:
  StoreType data[alloc_size];
  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  StackMemory() : stored_size(alloc_size) {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~StackMemory() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void map(StoreType* array, size_t array_max_size) {
    SINGLE_LAMBDA([&](){ stored_size = min(alloc_size, array_max_size); });
    WORKER_SYNC();

    VECTOR_LAMBDA(stored_size,
                  [&](size_t& i) {
                    data[i] = array[i];
                  });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void map(const amrex::Array4<StoreType>& array,
           const amrex::Dim3& lo, const amrex::Dim3& hi,
           const int src_comp = 0) {
    SINGLE_LAMBDA([&](){ stored_size = min(alloc_size, box_number_cells(lo, hi)); });
    WORKER_SYNC();

    VECTOR_BOX_LAMBDA(stored_size, lo, hi,
                      [&](size_t& i, size_t& j, size_t& k) {
                        data[i] = array[i];
                      });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void save(StoreType* array, size_t size_to_save = alloc_size) {
    size_t save_length = min(size_to_save, stored_size);
    VECTOR_LAMBDA(save_length,
                  [&](size_t& i) {
                    array[i] = data[i];
                  });
  }

};

template<class StoreType>
class HeapMemoryWindow {
public:
  StoreType* data;
  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  HeapMemoryWindow() : stored_size(0) {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~HeapMemoryWindow() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void map(StoreType* array, size_t array_max_size) {
    SINGLE_LAMBDA([&](){
                    stored_size = array_max_size;
                    data = array;
                  });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void save(StoreType* array, size_t array_max_size = 0) {}
};
#endif
