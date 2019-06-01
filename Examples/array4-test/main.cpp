#include <iostream>
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>

#include "VectorGpuMacros.H"
#include "RealVectorSet.H"
#include "VectorStorage.H"

using namespace amrex;

void initialize_domain_test(Real* fabptr,
                            const int lo_1, const int lo_2, const int lo_3,
                            const int hi_1, const int hi_2, const int hi_3,
                            const int tile_size_1, const int tile_size_2, const int tile_size_3,
                            const int tile_idx_1=0, const int tile_idx_2=0, const int tile_idx_3=0)
{
  const Dim3 begin = {lo_1, lo_2, lo_3};
  const Dim3 end = {hi_1+1, hi_2+1, hi_3+1};

  const Dim3 tile_size = {tile_size_1, tile_size_2, tile_size_3};

  PRINT_DIM3("f begin", begin);
  PRINT_DIM3("f end", end);
  PRINT_DIM3("f ts", tile_size);

  auto fabarray = Array4<Real>(fabptr, begin, end);

  std::cout << "fabarray stuff ..." << std::endl;
  PRINT_DIM3("begin", fabarray.begin);
  PRINT_DIM3("end", fabarray.end);
  std::cout << "jstride: " << fabarray.jstride << std::endl;
  std::cout << "kstride: " << fabarray.kstride << std::endl;
  std::cout << "nstride: " << fabarray.nstride << std::endl;

  Dim3 tile_lo, tile_hi;

  tile_lo.x = begin.x + tile_idx_1 * tile_size.x;
  tile_hi.x = tile_lo.x + tile_size.x - 1;

  tile_lo.y = begin.y + tile_idx_2 * tile_size.y;
  tile_hi.y = tile_lo.y + tile_size.y - 1;

  tile_lo.z = begin.z + tile_idx_3 * tile_size.z;
  tile_hi.z = tile_lo.z + tile_size.z - 1;

  PRINT_DIM3("f tlo", tile_lo);
  PRINT_DIM3("f thi", tile_hi);

  int icount = 0;
  for (int k = tile_lo.z; k <= tile_hi.z; k++) {
    for (int j = tile_lo.y; j <= tile_hi.y; j++) {        
      for (int i = tile_lo.x; i <= tile_hi.x; i++) {
        std::cout << "map (" << i << " " << j << " " << k << ") -> " << icount << std::endl;
        //fabarray(i,j,k) = icount; icount += 1;
        fabarray(i,j,k) = 100.0; icount += 1;
      }
    }
  }
}

template <size_t ncomp, size_t vector_length>
#ifdef AMREX_USE_CUDA
__global__
#endif
void initialize_domain(Real* fabptr,
                       const int lo_1, const int lo_2, const int lo_3,
                       const int hi_1, const int hi_2, const int hi_3,
                       const int tile_size_1, const int tile_size_2, const int tile_size_3,
                       const int tile_idx_1=0, const int tile_idx_2=0, const int tile_idx_3=0)
{
  const Dim3 lo = {lo_1, lo_2, lo_3};
  const Dim3 hi = {hi_1, hi_2, hi_3};
  const Dim3 tile_size = {tile_size_1, tile_size_2, tile_size_3};
  const Dim3 tile_idx = {tile_idx_1, tile_idx_2, tile_idx_3};

  PARALLEL_SHARED RealVectorSet<ncomp, vector_length,
                                FabWindow<Real>> fabvset;

  PARALLEL_SHARED RealVectorSet<ncomp, vector_length,
                                StackCreate<Real, vector_length>> scratch_vset;

  PARALLEL_REGION
    {
      fabvset.map(fabptr, lo, hi, tile_size, tile_idx, 0, 0, ncomp);
      WORKER_SYNC();

      scratch_vset = fabvset;
      scratch_vset += 2.0;
      scratch_vset -= 1.0;
      scratch_vset *= 4.0;
      scratch_vset /= 4.0;
      fabvset = scratch_vset;
      scratch_vset = 2.0;
      fabvset /= scratch_vset;
      fabvset *= scratch_vset;
      fabvset += scratch_vset;
      fabvset -= scratch_vset;
    }
}

void initialize_domain_ones(amrex::Array4<Real> fabarray,
                            const int lo_1, const int lo_2, const int lo_3,
                            const int hi_1, const int hi_2, const int hi_3,
                            const int ncomp = 1)
{
  const Dim3 lo = {lo_1, lo_2, lo_3};
  const Dim3 hi = {hi_1, hi_2, hi_3};

  for (int n = 0; n < ncomp; n++)
    for (int k = lo.z; k <= hi.z; k++)
      for (int j = lo.y; j <= hi.y; j++)
        for (int i = lo.x; i <= hi.x; i++)
          fabarray(i,j,k,n) = 1.0;
}

void print_domain(amrex::Array4<Real> fabarray,
                  const int lo_1, const int lo_2, const int lo_3,
                  const int hi_1, const int hi_2, const int hi_3)
{
  const Dim3 lo = {lo_1, lo_2, lo_3};
  const Dim3 hi = {hi_1, hi_2, hi_3};
  
  for (int k = lo.z; k <= hi.z; k++)
    for (int j = lo.y; j <= hi.y; j++)
      for (int i = lo.x; i <= hi.x; i++)
        Print() << "(" << i << ", " << j << ", " << k << "): " << fabarray(i,j,k) << "\n";
}

int main(int argc,
         char* argv[])
{
  Initialize(argc, argv);

  {
    int number_cells = 32;
    std::vector<int> ncells {AMREX_D_DECL(number_cells, number_cells, number_cells)};
    int max_grid_size = 16;

    // Read input parameters
    {
      ParmParse pp;
      if (!pp.queryarr("n_cells", ncells, 0, AMREX_SPACEDIM))
        Print() << "n_cells not specified, so using 32 cells in each dimension.\n";

      if (!pp.query("max_grid_size", max_grid_size))
        Print() << "max_grid_size not specified, so using 16.\n";
    }

    BoxArray ba;
    Geometry geo;

    // Define BoxArray and Geometry for our domain
    {
      // Define index space
      Box bx(IntVect(AMREX_D_DECL(0, 0, 0)),
             IntVect(AMREX_D_DECL(ncells[0]-1, ncells[1]-1, ncells[2]-1)),
             IntVect(AMREX_D_DECL(0, 0, 0)));
      ba.define(bx);
      ba.maxSize(max_grid_size);

      // Define physical space
      RealBox rbox(AMREX_D_DECL(0.0, 0.0, 0.0),
                   AMREX_D_DECL(1.0, 1.0, 1.0));

      // Cartesian coordinate system
      int coordinates = 0;

      // Fully periodic domain
      std::array<int,AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(1,1,1)};

      // Define Geometry
      geo.define(bx, &rbox, coordinates, is_periodic.data());
    }

    // Construct DistributionMapping
    DistributionMapping dm {ba};

    // 2 components, no ghost cells
    const int num_components = 2;
    const int num_ghost_cell = 0;

    // Build MultiFab
    MultiFab state(ba, dm, num_components, num_ghost_cell);

    // Initialize MultiFab on CPU to hold 1's
    for (MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      FArrayBox& fab = state[mfi];
      auto lo = bx.loVect3d();
      auto hi = bx.hiVect3d();
      auto fabarray = state.array(mfi);
      Print() << "nstride: " << fabarray.nstride << "\n";
      initialize_domain_ones(fabarray,
                             lo[0], lo[1], lo[2],
                             hi[0], hi[1], hi[2],
                             num_components);
    }

    {
      std::string pfname = "plt_" + std::to_string(ncells[0]);
#if (AMREX_SPACEDIM >= 2)
      pfname += "_" + std::to_string(ncells[1]);
#endif
#if (AMREX_SPACEDIM == 3)
      pfname += "_" + std::to_string(ncells[2]);
#endif
      pfname += "_" + std::to_string(max_grid_size);
      pfname += "_init";

      const Vector<std::string> varnames {"alpha", "beta"};

      WriteSingleLevelPlotfile(pfname, state, varnames, geo, 0.0, 0);
    }
   

    // Initialize MultiFab on GPU to hold 2's
    const size_t grid_size_per_block = 1024;
    IntVect cpu_tile_size(1024, 1, 1);

    for (MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto fabarray = state.array(mfi);
      
      auto lo = bx.loVect3d();
      auto hi = bx.hiVect3d();
      Print() << "lo: " << lo[0] << " " << lo[1] << " " << lo[2] << "\n";
      Print() << "hi: " << hi[0] << " " << hi[1] << " " << hi[2] << "\n";

      auto box_length = bx.length();
      Dim3 gpu_tile_size({box_length[0], box_length[1], box_length[2]});

      Dim3 gpu_num_blocks({1,1,1});
      int total_gpu_blocks = 1;

      auto tilelen = [&]()
                     { return static_cast<size_t>(gpu_tile_size.x * gpu_tile_size.y * gpu_tile_size.z); };

      Print() << "iterating\n";
      Print() << "box length: " << box_length[0] << " " << box_length[1] << " " << box_length[2] << "\n";      
      while(true) {
        Print() << "nBlocks: " << gpu_num_blocks.x << " " << gpu_num_blocks.y << " " << gpu_num_blocks.z << "\n";
        Print() << "tile size: " << gpu_tile_size.x << " " << gpu_tile_size.y << " " << gpu_tile_size.z << "\n";
        Print() << "grid size per block: " << grid_size_per_block << "\n";
        if ((gpu_tile_size.z % 2) == 0 &&
            box_length[2] % (gpu_tile_size.z / 2) == 0 &&
            tilelen() > grid_size_per_block) {
          gpu_tile_size.z /= 2;
          gpu_num_blocks.z *= 2;
          total_gpu_blocks *= 2;
          Print() << "continuing from x\n";
          continue;
        } else if ((gpu_tile_size.y % 2) == 0 &&
                   box_length[1] % (gpu_tile_size.y / 2) == 0 &&
                   tilelen() > grid_size_per_block) {                   
          gpu_tile_size.y /= 2;
          gpu_num_blocks.y *= 2;
          total_gpu_blocks *= 2;
          Print() << "continuing from y\n";          
          continue;
        } else if ((gpu_tile_size.x % 2) == 0 &&
                   box_length[0] % (gpu_tile_size.x / 2) == 0 &&
                   tilelen() > grid_size_per_block) {                   
          gpu_tile_size.x /= 2;
          gpu_num_blocks.x *= 2;
          total_gpu_blocks *= 2;
          Print() << "continuing from z\n";          
          continue;
        } else {
          Print() << "breaking\n";
          break;
        }
      }

      assert(tilelen() <= grid_size_per_block);

      Print() << "done iterating\n";

      Print() << "nBlocks: " << gpu_num_blocks.x << " " << gpu_num_blocks.y << " " << gpu_num_blocks.z << "\n";

      Print() << "Tile Size: " << gpu_tile_size.x << " " << gpu_tile_size.y << " " << gpu_tile_size.z << "\n";      

#ifndef AMREX_USE_CUDA      
      for (int kt = 0; kt < gpu_num_blocks.z; kt++) {
        for (int jt = 0; jt < gpu_num_blocks.y; jt++) {
          for (int it = 0; it < gpu_num_blocks.x; it++) {
      // for (int kt = 0; kt < 1; kt++) {
      //   for (int jt = 0; jt < 1; jt++) {
      //     for (int it = 0; it < 1; it++) {            
#else
            dim3 nBlocks(gpu_num_blocks.x, gpu_num_blocks.y, gpu_num_blocks.z);            
            dim3 nThreads(64,1,1);
            const int kt = 0;
            const int jt = 0;
            const int it = 0;
#endif

            initialize_domain<num_components, grid_size_per_block>
#ifdef AMREX_USE_CUDA
              <<<nBlocks, nThreads>>>
#endif
//            initialize_domain_test
              (fabarray.p,
               lo[0], lo[1], lo[2],
               hi[0], hi[1], hi[2],
               gpu_tile_size.x, gpu_tile_size.y, gpu_tile_size.z,
               it, jt, kt);

#ifndef AMREX_USE_CUDA
          }
        }
      }
#endif

    }

    if (false) {
    for (MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      FArrayBox& fab = state[mfi];
      auto lo = bx.loVect3d();
      auto hi = bx.hiVect3d();
      auto fabarray = state.array(mfi);
      print_domain(fabarray,
                   lo[0], lo[1], lo[2],
                   hi[0], hi[1], hi[2]);
    }
    }
    

    // Write MultiFab to plotfile
    std::string pfname = "plt_" + std::to_string(ncells[0]);
#if (AMREX_SPACEDIM >= 2)
    pfname += "_" + std::to_string(ncells[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    pfname += "_" + std::to_string(ncells[2]);
#endif
    pfname += "_" + std::to_string(max_grid_size);

    const Vector<std::string> varnames {"alpha", "beta"};

    WriteSingleLevelPlotfile(pfname, state, varnames, geo, 0.0, 0);

  }

  amrex::Finalize();
}
