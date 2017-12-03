#include <iostream>
#include <omp.h>
#include <cstdlib>

struct Matrix {
    int n;
    int m;
    int** data;
};

void genMatrix(Matrix &matr) {
    matr.data = new int*[matr.n];
    for (int i = 0; i < matr.n; ++i) {
        matr.data[i] = new int[matr.m];
    }

    for (int i = 0; i < matr.n; ++i) {
        for (int j = 0; j < matr.m; ++j) {
            matr.data[i][j] = rand();
        }
    }
}

Matrix transposeParallelMatr(const Matrix& matr) {
    Matrix matrRes;
    matrRes.n = matr.m;
    matrRes.m = matr.n;
    matrRes.data = new int*[matrRes.n];
    for (int i = 0; i < matrRes.n; ++i) {
        matrRes.data[i] = new int[matrRes.m];
    }

    double t1 = omp_get_wtime();
    int i, j;
#pragma omp parallel shared(matrRes)
{
#pragma omp for private(i, j)
    for (i = 0; i < matrRes.n; ++i) {
        for (j = 0; j < matrRes.m; ++j) {
            matrRes.data[i][j] = matr.data[j][i];
        }
    }
}
    double t2 = omp_get_wtime();

    printf("time of transpose: %g\n", t2 - t1);
    return matrRes;

}

//Matrix transposeMatr(const Matrix& matr) {
//    Matrix matrRes;
//    matrRes.n = matr.m;
//    matrRes.m = matr.n;
//    matrRes.data = new int*[matrRes.n];
//    for (int i = 0; i < matrRes.n; ++i) {
//        matrRes.data[i] = new int[matrRes.m];
//    }
//    double t1 = omp_get_wtime();
//    for (int i = 0; i < matrRes.n; ++i) {
//        for (int j = 0; j < matrRes.m; ++j) {
//            matrRes.data[i][j] = matr.data[j][i];
//        }
//    }
//    double t2 = omp_get_wtime();
//
//    printf("time of transpose without omp: %g\n", t2 - t1);
//    return matrRes;
//}

void freeMatrix(Matrix& matr) {
    for (int i = 0; i < matr.n; ++i) {
        delete[] matr.data[i];
    }
    delete[] matr.data;
}

int main(int argc, char* argv[]) {
    size_t threadCount = atoi(argv[1]);    // threads count
    omp_set_num_threads(threadCount);
    omp_set_dynamic(0);

    if (threadCount <= 0) {
        printf("thread count must be > 0");
        return 0;
    }

    Matrix matr;

    matr.n = atoi(argv[2]);    // matrix rows count
    matr.m = atoi(argv[3]);    // matrix cols count
    if (matr.n <= 0 || matr.m <= 0) {
        printf("matrix size is empty\n");
        return 0;
    }

    genMatrix(matr);
    if (matr.n <= 0 || matr.m <= 0) {
        printf("empty matrix\n");
        return 0;
    }
    Matrix matrRes = transposeParallelMatr(matr);
    freeMatrix(matrRes);
    freeMatrix(matr);
    return 0;
}