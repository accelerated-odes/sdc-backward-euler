#include <iostream>
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>

#include "VectorParallelUtil.H"
#include "VectorStorage.H"
#include "RealVectorSet.H"

using namespace amrex;

template <size_t ncomp, size_t vector_length>
#ifdef AMREX_USE_CUDA
__global__
#endif
void do_some_math(Real* fabptr,
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

void initialize_ones(amrex::Array4<Real> fabarray,
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
      initialize_ones(fabarray,
                      lo[0], lo[1], lo[2],
                      hi[0], hi[1], hi[2],
                      num_components);
    }

    // Write initial state to plotfile
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

    // Do some math in shared memory
    const size_t grid_size_per_block = 1024;

    for (MFIter mfi(state, false); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto fabarray = state.array(mfi);

      Dim3 kernel_tile_size, kernel_num_blocks;

      get_tiling(bx, grid_size_per_block, kernel_tile_size, kernel_num_blocks);
      
      auto lo = bx.loVect3d();
      auto hi = bx.hiVect3d();

#ifndef AMREX_USE_CUDA      
      for (int kt = 0; kt < kernel_num_blocks.z; kt++) {
        for (int jt = 0; jt < kernel_num_blocks.y; jt++) {
          for (int it = 0; it < kernel_num_blocks.x; it++) {
#else
            dim3 nBlocks(kernel_num_blocks.x, kernel_num_blocks.y, kernel_num_blocks.z);
            dim3 nThreads(64,1,1);
            const int kt = 0;
            const int jt = 0;
            const int it = 0;
#endif

            do_some_math<num_components, grid_size_per_block>
#ifdef AMREX_USE_CUDA
              <<<nBlocks, nThreads>>>
#endif
              (fabarray.p,
               lo[0], lo[1], lo[2],
               hi[0], hi[1], hi[2],
               kernel_tile_size.x, kernel_tile_size.y, kernel_tile_size.z,
               it, jt, kt);

#ifndef AMREX_USE_CUDA
          }
        }
      }
#endif

    }

    // Write final state to plotfile
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
