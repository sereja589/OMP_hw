#include <iostream>
#include <mpi.h>
#include <cstdlib>

struct Matrix {
    int n;
    int m;
    int* data;
};

void genMatrix(Matrix& matr) {
    matr.data = new int[matr.n * matr.m];
    for (int i = 0; i < matr.n * matr.m; ++i) {
        matr.data[i] = rand();
    }
}

void transposeMatr(const Matrix& matr, Matrix& matRes, int blockBegin, int blockSize) {
    for (int cur = blockBegin; cur < blockBegin + blockSize; ++cur) {
        int i = cur / matr.m;
        int j = cur % matr.m;
        matRes.data[j * matRes.m + i] = matr.data[cur];
    }
}

void freeMatrix(Matrix& matr) {
    delete[] matr.data;
}

int main(int argc, char* argv[]) {
    int error;

    if ((error = MPI_Init(&argc, &argv)) != MPI_SUCCESS) {
        std::cerr << "error in MPI_Init" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error);
    }

    int procNum, procCount;

    if ((error = MPI_Comm_size(MPI_COMM_WORLD, &procCount)) != MPI_SUCCESS) {
        std::cerr << "error in MPI_Comm_size" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error);
    }
    if ((error = MPI_Comm_rank(MPI_COMM_WORLD, &procNum)) != MPI_SUCCESS) {
        std::cerr << "error in MPI_Comm_rank" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error);
    }

    Matrix matr;

    matr.n = atoi(argv[1]);    // matrix rows count
    matr.m = atoi(argv[2]);    // matrix cols count

    if (matr.n <= 0 || matr.m <= 0) {
        printf("matrix size is empty\n");
        return 0;
    }
    
    if (procNum == 0)
        genMatrix(matr);

    MPI_Bcast(matr.data, matr.n * matr.m, MPI_INT, 0, MPI_COMM_WORLD);

    Matrix matrRes;
    matrRes.n = matr.m;
    matrRes.m = matr.n;
    matrRes.data = new int[matr.n * matr.m];
    std::fill(matrRes.data, matrRes.data + matrRes.n * matrRes.m, 0);

    Matrix matrReduce;
    matrReduce.n = matrRes.n;
    matrReduce.m = matrRes.m;
    matrReduce.data = new int[matr.n * matr.m];
    std::fill(matrReduce.data, matrReduce.data + matrReduce.n * matrReduce.m, 0);

    int matrSz = matr.n * matr.m;

    int blockSize = matrSz / procCount;
    int restCeils = matrSz % procCount;

    int curBlockBegin = blockSize * procNum + std::min(procNum, restCeils);
    int curBlockSize = blockSize + int(restCeils > procNum);

    MPI_Barrier(MPI_COMM_WORLD);

    double timeStart = MPI_Wtime();

    transposeMatr(matr, matrRes, curBlockBegin, curBlockSize);

    MPI_Barrier(MPI_COMM_WORLD);

    if (procNum == 0) {
        std::cout << "time: " << MPI_Wtime() - timeStart << std::endl;
//        for (int i = 0; i < matrReduce.n; ++i) {
//            for (int j = 0; j < matrReduce.m; ++j) {
//                std::cout << matrReduce.data[i * matrReduce.m + j] << ' ';
//            }
//            std::cout << std::endl;
//        }
    }

    MPI_Reduce(matrRes.data, matrReduce.data, matr.n * matr.m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    freeMatrix(matr);
    freeMatrix(matrRes);
    freeMatrix(matrReduce);

    if ((error = MPI_Finalize()) != MPI_SUCCESS) {
        std::cerr << "MPI_Finalize" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error);
    }

    return 0;
}
