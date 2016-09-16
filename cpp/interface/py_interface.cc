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

#include <boost/python.hpp>
#include <typeinfo>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/gpu_operations.h"
#include "include/svd_solver.h"
#include "include/gpu_svd_solver.h"
#include "include/util.h"
#include "include/gpu_util.h"

#include "interface/py_interface.h"

namespace Nice {

PyInterface::PyInterface()
: dtype_(DOUBLE),
input_imat_(nullptr, 0, 0),
input_fmat_(nullptr, 0, 0),
input_dmat_(nullptr, 0, 0),
output_imat_(nullptr, 0, 0),
output_fmat_(nullptr, 0, 0),
output_dmat_(nullptr, 0, 0) {}

PyInterface::PyInterface(DataType dtype)
: dtype_(dtype),
input_imat_(nullptr, 0, 0),
input_fmat_(nullptr, 0, 0),
input_dmat_(nullptr, 0, 0),
output_imat_(nullptr, 0, 0),
output_fmat_(nullptr, 0, 0),
output_dmat_(nullptr, 0, 0) {}

void PyInterface::Init(const char *path, int row, int col) {
  // Create corresponding matrix/vector based on the given path
}

void PyInterface::Init(PyObject *in, int row, int col) {
  // Get the python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(in, &pybuf, PyBUF_SIMPLE);

  // Initialze the input data
  switch (dtype_) {
    case FLOAT:
      new (&input_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                    row, col);
      break;
    case DOUBLE:
      new (&input_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                    row, col);
      break;
    default:
      break;
  }
}

void PyInterface::SetupParams(const boost::python::dict &params,
                              ModelType model_type) {
}

void PyInterface::Run(ModelType model_type, PyObject *out, int row, int col) {
  // Get python object buffer
  Py_buffer pybuf;
  PyObject_GetBuffer(out, &pybuf, PyBUF_SIMPLE);

  // Get parameters
  boost::python::dict param;
  if (param_map_.find(model_type) != param_map_.end())
    param = param_map_[model_type];

  switch (dtype_) {
    case FLOAT:
      new (&output_fmat_) FMatrixMap(reinterpret_cast<float *>(pybuf.buf),
                                     row, col);
      break;
    case DOUBLE:
      new (&output_dmat_) DMatrixMap(reinterpret_cast<double *>(pybuf.buf),
                                     row, col);
      break;
    default:
      break;
  }

  // Run the KMEANS model
  if (model_type == KMEANS) {
    if (dtype_ == FLOAT) {
      output_fmat_ = RunKmeans<float>(param,
                input_fmat_);
    } else if (dtype_ == DOUBLE) {
      output_dmat_ = RunKmeans<double>(param,
                input_dmat_);
    }
  }
  // Run Inverse only for test
  if (model_type == INVERSE) {
    if (dtype_ == FLOAT) {
      output_fmat_ = RunInverse<float>(input_fmat_);
    } else if (dtype_ == DOUBLE) {
      output_dmat_ = RunInverse<double>(input_dmat_);
    }
  }
}

}  // namespace Nice


void (Nice::PyInterface::*Init1)(const char *, int row, int col)
  = &Nice::PyInterface::Init;
void (Nice::PyInterface::*Init2)(PyObject *, int row, int col)
  = &Nice::PyInterface::Init;

BOOST_PYTHON_MODULE(Nice4Py) {
  boost::python::enum_<Nice::DataType>("DataType")
    .value("INT", Nice::DataType::INT)
    .value("FLOAT", Nice::DataType::FLOAT)
    .value("DOUBLE", Nice::DataType::DOUBLE);
  boost::python::enum_<Nice::ModelType>("ModelType")
    .value("KMEANS", Nice::ModelType::KMEANS)
    .value("INVERSE", Nice::ModelType::INVERSE)
    .value("OTHERS", Nice::ModelType::OTHERS);
  boost::python::class_<Nice::PyInterface>("PyInterface",
                                           boost::python::init<>())
    .def(boost::python::init<Nice::DataType>())
    .def("Init", Init1)
    .def("Init", Init2)
    .def("SetupParams", &Nice::PyInterface::SetupParams)
    .def("Run", &Nice::PyInterface::Run);
}
