

nvcc kernel.cu -c -std=c++11 -O3 -D_FORCE_INLINES &&\
g++ test.cc matmul.cpp kernel.o -o test -lgtest -lgtest_main -std=c++11 -lpthread -fopenmp -lcudart -O3 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 &&\
rm kernel.o && \
echo "compile finished!"
