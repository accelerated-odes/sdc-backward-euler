from SparseGaussJordan import GaussJordan
import argparse

if __name__=='__main__':
    ## Read CMD Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('maskfile', type=str,
                        help='Name of the input mask file for the array A in A*x=b.')
    parser.add_argument('-csr', action='store_true',
                        help='Use compressed sparse row (CSR) matrix format.')
    parser.add_argument('-py', type=str,
                        help='Name of the Python output file to generate.')
    parser.add_argument('-f95', type=str,
                        help='Name of the Fortran-95 output file to generate.')
    parser.add_argument('-cpp', type=str,
                        help='Name of the C++ output file to generate.')
    parser.add_argument('-smp', action='store_true',
                        help='Attempt to simplify solution. Can be very slow, but if possible, will reduce the number of operations required for the solution.')
    parser.add_argument('-expand', action='store_true',
                        help='Simplify resulting expressions using Sympy expand()')
    parser.add_argument('-cse', action='store_true',
                        help='Execute Common Subexpression Elimination. (After simplification if the -smp option is present.) This is pretty fast.')
    parser.add_argument('-v', action='store_true',
                        help='Enable verbose output.')
    args = parser.parse_args()

    GJ = GaussJordan(structure_file=args.maskfile, compressed_sparse_row=args.csr, out_py=args.py, out_f95=args.f95, out_cpp=args.cpp, smp=args.smp, expand=args.expand, cse=args.cse, verbose=args.v)
