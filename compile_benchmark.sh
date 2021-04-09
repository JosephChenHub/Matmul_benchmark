

python3 gen_bench_code.py &&\
nvcc kernel.cu -c -std=c++11 -O3  -D_FORCE_INLINES -arch=sm_72&&\
g++ kernel.o matmul.cpp benchmark.cpp -o bench -std=c++11 -lpthread -fopenmp -lbenchmark -lcudart -lcublas -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -march=native -mavx512f -O3 -Wall&&\
rm kernel.o && \
echo "compile finished!"
