// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "include/cpu_operations.h"
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

// This function returns the transpose of a matrix
template<typename T>
Matrix<T> CpuOperations<T>::Transpose(const Matrix<T> &a) {
  return a.transpose();  // Return transpose
}

template<typename T>
Vector<T> CpuOperations<T>::Transpose(const Vector<T> &a) {
  return a.transpose();
}


// Returns the resulting matrix that is created by running a logical or
// operation on the two input matrices
template<typename T>
Matrix<bool> CpuOperations<T>::LogicalOr(const Matrix<bool> &a,
                                        const Matrix<bool> &b) {
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
    std::cerr << std::endl << "ERROR: MATRICES ARE NOT THE SAME SIZE!"
    << std::endl << std::endl;
      exit(1);  // Exits the program
  } else if (b.rows() == 0 || b.cols() == 0 || a.rows() == 0 || a.cols() == 0) {
    std::cerr << std::endl << "ERROR: EMPTY MATRIX AS ARGUMENT!"
    << std::endl << std::endl;
    exit(1);  // Exits the program
  }
  return (a.array() || b.array());
}

// Returns the resulting vector that is created by running a logical or
// operation on the two input vectors
template<typename T>
Vector<bool> CpuOperations<T>::LogicalOr(const Vector<bool> &a,
                                        const Vector<bool> &b) {
  if ( a.size() != b.size() ) {
    std::cerr << std::endl << "ERROR: VECTORS ARE NOT THE SAME SIZE!"
    << std::endl << std::endl;
    exit(1);  // Exits the program
  } else if (a.size() == 0 || b.size() == 0) {
    std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!"
    << std::endl << std::endl;
    exit(1);  // Exits the program
  }
  return (a.array() || b.array());
}

template<typename T>
Matrix<bool> CpuOperations<T>::LogicalNot(const Matrix<bool> &a) {
  Matrix<bool> b = a.replicate(1, 1);
  int r;
  // Iterate through the copied matrix
  for (r = 0; r < b.rows(); ++r) {
    for (int c = 0; c < b.cols(); ++c) {
      b(r, c) = !b(r, c);
    }
  }
  if (b.rows() == 0 || b.cols() == 0) {
    std::cerr << std::endl << "ERROR: EMPTY MATRIX AS ARGUMENT!"
    << std::endl << std::endl;
    exit(1);  // Exits the program
  }
  return b;
}

template<typename T>
Vector<bool> CpuOperations<T>::LogicalNot(const Vector<bool> &a) {
  Vector<bool> b = a.replicate(1, 1);
  int i;
  // Iterate through vector
  for (i = 0; i < b.size(); ++i) {
    b(i) = !b(i);
  }
  if (a.size() == 0) {
    std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!"
    << std::endl << std::endl;
    exit(1);  // Exits the program
    }
  return b;
}

// Scalar-matrix multiplication
template<typename T>
Matrix<T> CpuOperations<T>::Multiply(const Matrix<T> &a, const T &scalar) {
    return scalar * a;
}

// Matrix-matrix multiplication
template<typename T>
Matrix<T> CpuOperations<T>::Multiply(const Matrix<T> &a, const Matrix<T> &b) {
    return a * b;
}

// Trace of a matrix
template<typename T>
T CpuOperations<T>::Trace(const Matrix<T> &a) {
    return a.trace();
}

/*
// Rank of a matrix
template<typename T>
T CpuOperations<T>::Rank(const Matrix<T> &a) {
    return a.rank();
}
*/


// This function returns the logical AND of two boolean matrices
template<typename T>
Matrix<bool> CpuOperations<T>::LogicalAnd(const Matrix<bool> &a,
                                          const Matrix<bool> &b) {
  // Checks to see that the number of rows and columns are the same
  if ((a.rows() != b.rows()) || (a.cols() != b.cols())) {
    std::cerr << "/nERROR: MATRICES ARE NOT THE SAME SIZE!/n/n";
    exit(1);  // Exits the program
  }
  return (a.array() && b.array());
  // Will return a matrix due to implicit conversion
}


// Returns the frobenius norm of the matrix
template<typename T>
T CpuOperations<T>::FrobeniusNorm(const Matrix<T> &a) {
  if (a.rows() == 0 || a.cols() == 0) {
    std::cout << std::endl << "ERROR: EMPTY MATRIX AS ARGUMENT!"
    << std::endl << std::endl;
    exit(-1);  // Exits the program
  } else {
    return a.norm();
  }
}

// This function returns the outer product of he two passed in vectors
template<typename T>
Matrix<T> CpuOperations<T>::OuterProduct(const Vector<T> &a,
                                         const Vector<T> &b) {
  if (a.size() == 0 || b.size() == 0) {
    std::cerr << std::endl << "ERROR: EMPTY VECTOR AS ARGUMENT!" << std::endl << std::endl;
    exit(1);
  }
  return a * b.transpose();
}

template class CpuOperations<int>;
template class CpuOperations<float>;
template class CpuOperations<double>;
template class CpuOperations<bool>;

}  //  namespace Nice

