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


#include "include/gpu_operations.h"
#include "Eigen/Dense"
#include "gtest/gtest.h"

template<class T>
class GpuMatrixMatrixAddTest : public ::testing::Test {
 public:
  Nice::Matrix<T> a;
  Nice::Matrix<T> b;
  Nice::Matrix<T> correct_ans;
  Nice::Matrix<T> calc_ans;

  void Add() {
    calc_ans = Nice::GpuOperations<T>::Add(a, b);
  }
};

typedef ::testing::Types<float, double> dataTypes;
TYPED_TEST_CASE(GpuMatrixMatrixAddTest, dataTypes);

TYPED_TEST(GpuMatrixMatrixAddTest, BasicTest) {
  this->a.resize(3, 3);
  this->b.resize(3, 3);
  this->correct_ans.resize(3, 3);
  this->a << 0.0, 1.0, 0.0,
             1.0, 0.0, 1.0,
             0.0, 1.0, 0.0;

  this->b << 1.0, 0.0, 1.0,
             0.0, 1.0, 0.0,
             1.0, 0.0, 1.0;

  this->correct_ans << 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0;
  this->Add();
  ASSERT_TRUE(this->correct_ans.isApprox(this->calc_ans));
}
