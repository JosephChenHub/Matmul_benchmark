



nvcc kernel.cu -c -std=c++11 -O3 -D_FORCE_INLINES -arch=sm_72 &&\
g++ test.cc matmul.cpp kernel.o -o test -lgtest -lgtest_main -std=c++11 -lpthread -fopenmp -lcudart -lcublas -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -march=native -O3&&\
rm kernel.o && \
echo "compile finished!"
