#ifndef _VECTOR_STORAGE_H
#define _VECTOR_STORAGE_H

#include "AMReX_Array.H"
#include "VectorParallelUtil.H"

#ifndef AMREX_USE_CUDA
using std::min;
using std::max;
#endif

template<typename StoreType, size_t alloc_size>
class StackCreate {
public:
  StoreType data[alloc_size];
  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  StackCreate() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~StackCreate() {}

  template<typename AnyIndexType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  StoreType& operator[](AnyIndexType i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void resize(size_t new_size = alloc_size) {
    SINGLE_LAMBDA([&](){
                    stored_size = new_size;
                  });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void map(StoreType* array, size_t array_max_size) {
    size_t map_length = min(array_max_size, alloc_size);

    VECTOR_LAMBDA(map_length,
                  [&](size_t& i) {
                    data[i] = array[i];
                  });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void save(StoreType* array, size_t size_to_save = alloc_size) {
    size_t save_length = min(size_to_save, alloc_size);

    VECTOR_LAMBDA(save_length,
                  [&](size_t& i) {
                    array[i] = data[i];
                  });
  }

};

template<class StoreType, size_t null_param>
class HeapWindow {
public:
  StoreType* data;
  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  HeapWindow() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~HeapWindow() {}

  template<typename AnyIndexType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  StoreType& operator[](AnyIndexType i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void resize(size_t new_size = 0) {
    SINGLE_LAMBDA([&](){
                    stored_size = new_size;
                  });
  }

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

template<class StoreType, size_t null_param>
class FabWindow {
public:
  StoreType* data;

  amrex::Dim3 fab_begin, fab_end; // begin, end for the Fab Array4
                                  // note that end = hi + 1

  amrex::Dim3 tile_lo, tile_hi; // lo, hi for our tile of the Fab

  amrex::Dim3 tile_dim; // dimensions of our tile of the Fab

  size_t stored_size;

  int comp;    // this is which component of the Fab we map to

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  FabWindow() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~FabWindow() {}

  template<typename AnyIndexType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  StoreType& operator[](AnyIndexType i) {
    // we map the index i into our lo, hi region to get indices
    // to access the Fab (idx, jdx, kdx).

    // this could be optimized ... perhaps by storing the Array4?

    const int ztile_size = tile_dim.x * tile_dim.y;

    const int kdx = static_cast<int>(i) / ztile_size + tile_lo.z;
    const int xy_index = static_cast<int>(i) % ztile_size;

    const int jdx = xy_index / tile_dim.x + tile_lo.y;
    const int idx = xy_index % tile_dim.x + tile_lo.x;

    const auto array = amrex::Array4<StoreType>(data, fab_begin, fab_end);

    return array(idx, jdx, kdx, comp);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void resize(size_t new_size = 0) {
    SINGLE_LAMBDA([&](){
                    stored_size = new_size;
                  });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void map(StoreType* fabptr,
           const amrex::Dim3& fab_lo, const amrex::Dim3& fab_hi,
           const amrex::Dim3& tile_size, const amrex::Dim3& tile_idx,
           const int src_comp = 0) {
    data = fabptr;
    comp = src_comp;

    fab_begin.x = fab_lo.x; fab_begin.y = fab_lo.y; fab_begin.z = fab_lo.z;
    fab_end.x = fab_hi.x + 1; fab_end.y = fab_hi.y + 1; fab_end.z = fab_hi.z + 1;

    // tile_size is a Dim3 containing the number of cells along
    // each dimension defining the shape of each tile.

    // NOTE: it is assumed that the tile size passed to this routine
    // is evenly divisible into the Fab dimensions so that
    // the number of worker groups along each dimension
    // is equal to the number of tiles along the corresponding dimension.

    tile_lo.x = fab_lo.x + PARALLEL_TILE_INDEX_X(tile_idx.x) * tile_size.x;
    tile_hi.x = tile_lo.x + tile_size.x - 1;

    tile_lo.y = fab_lo.y + PARALLEL_TILE_INDEX_Y(tile_idx.y) * tile_size.y;
    tile_hi.y = tile_lo.y + tile_size.y - 1;

    tile_lo.z = fab_lo.z + PARALLEL_TILE_INDEX_Z(tile_idx.z) * tile_size.z;
    tile_hi.z = tile_lo.z + tile_size.z - 1;

    tile_dim.x = tile_size.x; tile_dim.y = tile_size.y; tile_dim.z = tile_size.z;

    stored_size = static_cast<size_t>(tile_dim.x * tile_dim.y * tile_dim.z);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void save(StoreType* array, size_t array_max_size = 0) {
    // We don't actually do anything here because we operate
    // on Fab data in place but we provide this stub
    // to support the same API as other storage types.
  }
};
#endif
