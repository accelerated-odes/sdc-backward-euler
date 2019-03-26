#ifndef SPARSE_LINEAR_SOLVER_H
#define SPARSE_LINEAR_SOLVER_H
#include <iostream>
#include <cusolverSp.h>
#include <cusparse.h>
#include "IntegerVector.H"

template<class SystemClass> class SparseLinearSolver {
    // Solver
    size_t workspace_size, internal_size;
    cusparseMatDescr_t system_description;
    cusolverSpHandle_t cusolverHandle;
    cusparseHandle_t cusparseHandle;
    csrqrInfo_t info;
    void*  workspace;

    // System
    int* matrices_csr_col_index_d;
    int* matrices_csr_row_count_d;
    int* matrices_csr_col_index_h; //[SystemClass::sparse_jac_nnz];
    int* matrices_csr_row_count_h; //[SystemClass::neqs+1];
    double* matrices_csr_values;
    double* system_x;
    double* system_b;

    int number_batched_systems;
private:
    void create_linear_solver(size_t number_systems) {
        number_batched_systems = static_cast<int>(number_systems);
        
        // initialize the linear solver memory
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

        // Make handle for cuSolver
#if VERBOSE_DEBUG
        std::cout << "Creating cuSolver Handle" << std::endl;
#endif

        cusolver_status = cusolverSpCreate(&cusolverHandle);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        // Create an info object
#if VERBOSE_DEBUG
        std::cout << "Creating info object" << std::endl;
#endif

        cusolver_status = cusolverSpCreateCsrqrInfo(&info);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        // Setup Sparse system description
#if VERBOSE_DEBUG
        std::cout << "Creating Matrix Descriptor" << std::endl;
#endif

        cusparse_status = cusparseCreateMatDescr(&system_description);
        assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if VERBOSE_DEBUG
        std::cout << "In Matrix Descriptor, setting Matrix Type" << std::endl;
#endif
        cusparse_status = cusparseSetMatType(system_description, CUSPARSE_MATRIX_TYPE_GENERAL);
        assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if VERBOSE_DEBUG
        std::cout << "In Matrix Descriptor, setting Matrix Index Base" << std::endl;
#endif
        cusparse_status = cusparseSetMatIndexBase(system_description, CUSPARSE_INDEX_BASE_ONE);
        assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if VERBOSE_DEBUG
        std::cout << "Created CV_cuSolver_Mem object." << std::endl;
#endif

        // analyze matrix system
        cusolver_status = CUSOLVER_STATUS_SUCCESS;

        cusolver_status = cusolverSpXcsrqrAnalysisBatched(cusolverHandle,
                                                          SystemClass::neqs, // size per subsystem
                                                          SystemClass::neqs, // size per subsystem
                                                          SystemClass::nnz,
                                                          system_description,
                                                          matrices_csr_row_count_d,
                                                          matrices_csr_col_index_d,
                                                          info);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cusolver_status = cusolverSpDcsrqrBufferInfoBatched(cusolverHandle,
                                                            SystemClass::neqs, // size per subsystem
                                                            SystemClass::neqs, // size per subsystem
                                                            SystemClass::nnz,
                                                            system_description,
                                                            matrices_csr_values,
                                                            matrices_csr_row_count_d,
                                                            matrices_csr_col_index_d,
                                                            number_systems,
                                                            info,
                                                            &internal_size,
                                                            &workspace_size);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaError_t cuda_status = cudaMalloc((void**) &workspace, workspace_size);
        assert(cuda_status == cudaSuccess);
    }

    void destroy_linear_solver() {
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cuda_status = cudaSuccess;

        cusolver_status = cusolverSpDestroy(cusolverHandle);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cusolver_status = cusolverSpDestroyCsrqrInfo(info);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cuda_status = cudaFree(workspace);
        assert(cuda_status == cudaSuccess);
    }
public:
    SparseLinearSolver(size_t number_systems, const SystemClass& ode_system) {
        // initialize batched matrix memory
        cudaError_t cuda_status = cudaSuccess;

        cuda_status = cudaMalloc((void**) &matrices_csr_values,
                                 sizeof(double)*SystemClass::nnz * number_systems);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMalloc((void**) &system_x,
                                 sizeof(double)*SystemClass::neqs * number_systems);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMalloc((void**) &system_b,
                                 sizeof(double)*SystemClass::neqs * number_systems);
        assert(cuda_status == cudaSuccess);

        // initialize the CSR matrix data
        IntegerVector<SystemClass::nnz> col_index = ode_system.csr_col_index;
        for (size_t i = 0; i < SystemClass::nnz; i++) {
            matrices_csr_col_index_h[i] = col_index[i];
        }

        IntegerVector<SystemClass::neqs+1> row_count = ode_system.csr_row_count;
        for (size_t i = 0; i < SystemClass::neqs+1; i++) {
            matrices_csr_row_count_h[i] = row_count[i];
        }

        // Allocate device memory and copy system parameters from host
        cuda_status = cudaMalloc((void**) &matrices_csr_col_index_d,
                                 sizeof(int) * SystemClass::nnz);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMalloc((void**) &matrices_csr_row_count_d,
                                 sizeof(int) * (SystemClass::neqs+1));
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMemcpy(matrices_csr_col_index_d, matrices_csr_col_index_h,
                                 sizeof(int) * SystemClass::nnz,
                                 cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMemcpy(matrices_csr_row_count_d, matrices_csr_row_count_h,
                                 sizeof(int) * (SystemClass::neqs+1),
                                 cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess);

        create_linear_solver(number_systems);
    }

    ~SparseLinearSolver() {
        destroy_linear_solver();
        
        // free batched matrix memory
        cudaError_t cuda_status = cudaSuccess;

        cuda_status = cudaFree(matrices_csr_values);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaFree(system_x);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaFree(system_b);
        assert(cuda_status == cudaSuccess);

        // free CSR matrix data
        cuda_status = cudaFree(matrices_csr_col_index_d);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaFree(matrices_csr_row_count_d);
        assert(cuda_status == cudaSuccess);
    }

    double* get_device_matrix() {
        // Return the device pointer to the batched matrix memory
        return matrices_csr_values;
    }

    double* get_device_vector() {
        // Return the device pointer to the batched vector rhs
        return system_b;
    }

    double* get_device_solution() {
        // Return the device pointer to the solution vector
        return system_x;
    }

    void solve() {
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

        // do the solve
        cusolver_status = cusolverSpDcsrqrsvBatched(cusolverHandle,
                                                    SystemClass::neqs, // size per subsystem
                                                    SystemClass::neqs, // size per subsystem
                                                    SystemClass::nnz,
                                                    system_description,
                                                    matrices_csr_values,
                                                    matrices_csr_row_count_d,
                                                    matrices_csr_col_index_d,
                                                    system_b,
                                                    system_x,
                                                    number_batched_systems,
                                                    info,
                                                    workspace);
        cudaError_t cuda_status = cudaDeviceSynchronize();
        assert(cuda_status == cudaSuccess);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    }
};
#endif
