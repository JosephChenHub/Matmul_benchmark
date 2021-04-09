

def gen_headers():
    dst = '#include <benchmark/benchmark.h>\n'
    dst += '#include "matmul.hpp"\n\n'

    return dst


def gen_code(name, method, lo, hi, trans=False):
    func_name = "BM_%s"%name
    func_name = func_name.replace("-", "_")
    dst = 'static void %s(benchmark::State& state) {\n'% func_name
    dst += '\tconst int n = static_cast<int>(state.range(0));\n'
    dst += '\tauto A = Matrix<float>::randn(n, n);\n'
    dst += '\tauto B = Matrix<float>::randn(n, n);\n'
    dst += '\tauto C = Matrix<float>::zeros(n, n);\n'
    if 'cuda' in method or 'CUDA' in method or 'cublas' in method:
        dst += '\t A = A.cuda(); B = B.cuda(); C = C.cuda();\n'

    dst += '\n\tfor(auto _ : state) {\n'
    dst += '\t\tauto start = std::chrono::high_resolution_clock::now();\n'
    if trans:
        dst += '\t\t%s(A, B.transpose(), C);\n'%method
    else:
        dst += '\t\t%s(A, B, C);\n'%method
    if 'cuda' in method or 'CUDA' in method or 'cublas' in method:
        dst += '\t\tcudaDeviceSynchronize();\n'
    dst += '\t\tauto end = std::chrono::high_resolution_clock::now();\n'
    dst += '\t\tauto elapsed_seconds = \n\t\t\t'
    dst += 'std::chrono::duration_cast<std::chrono::duration<double>>(end - start);\n'
    dst += '\t\tstate.SetIterationTime(elapsed_seconds.count());\n'
    dst += '\t}\n'
    dst += '\tstate.SetComplexityN(state.range(0));\n'
    dst += '}\n'

    dst += 'BENCHMARK(%s)->Name("%s")\n'%(func_name, name)
    dst += '\t\t->RangeMultiplier(2)->Range(%s, %s)\n'%(lo, hi)
    dst += '\t\t->Complexity(benchmark::oN)\n'
    dst += '\t\t->UseManualTime()\n'
    dst += '\t\t->Unit(benchmark::kMicrosecond);\n\n'

    return dst


dst = gen_headers()
dst += gen_code('cpu-naive', 'matmul_naive', 8, 8<<8)
dst += gen_code('cpu-trans', 'matmul_trans', 8, 8<<8, True)
dst += gen_code('cpu-trans-block', 'matmul_trans_block', 8, 8<<8, True)
dst += gen_code('cpu-openmp', 'matmul_openmp', 8, 8<<8)
dst += gen_code('cpu-omp-sse', 'matmul_omp_sse', 8, 8<<9, True)
dst += gen_code('cpu-omp-avx', 'matmul_omp_avx', 8, 8<<9, True)
dst += gen_code('cpu-omp-avx512', 'matmul_omp_avx512', 8, 8<<9, True)
dst += gen_code('cuda-naive', 'matmul_cuda_naive', 8, 8<<10)
dst += gen_code('cuda-tile', 'matmul_cuda_tile', 8, 8<<10)
dst += gen_code('cuda-unroll', 'matmul_cuda_unroll', 8, 8<<10)
dst += gen_code("cublas-sgemm", "matmul_cublas", 8, 8<<10)
#dst += gen_code("cuda-vectorize", "matmul_cuda_vectorize", 8<<9, 8<<10)

dst += "\nBENCHMARK_MAIN();\n"
#print(dst)

with open("benchmark.cpp", "w") as fout:
    fout.writelines(dst)
