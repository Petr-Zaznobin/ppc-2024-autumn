#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::get_func(
    const std::function<double(double)>& func) {
  f = func;
}

void zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::get_func(
    const std::function<double(double)>& func) {
  f = func;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  n = *reinterpret_cast<int*>(taskData->inputs[2]);

  results_.resize(1, 0.0);

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() < 3) {
    return false;
  }

  double a_val = *reinterpret_cast<double*>(taskData->inputs[0]);
  double b_val = *reinterpret_cast<double*>(taskData->inputs[1]);
  if (a_val >= b_val) {
    return false;
  }

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  double width = (b - a) / n;
  input_.resize(n);
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = a + i * width;
    sum += f(x) * width;
  }
  results_[0] = sum;

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = results_[0];

  return true;
}

double zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::integrate(
    const std::function<double(double)>& f_, double a_, double b_, int n_) {
  int rank = world.rank();
  int size = world.size();

  double width = (a - b) / n;
  int local_num_intervals = n_ / size;
  int remainder = n_ % size;

  if (rank < remainder) {
    local_num_intervals = local_num_intervals + 1;
  }

  double local_start = a_ + rank * local_num_intervals * width;

  double local_sum = 0.0;
  for (int i = 0; i < local_num_intervals; ++i) {
    double x = local_start + i * width;
    local_sum += f(x) * width;
  }

  return local_sum;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int d = 0;
  if (world.rank() == 0) {
    d = n / world.size();
  }
  MPI_Bcast(&d, 1, MPI_UNSIGNED, 0, world);

  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    n = *reinterpret_cast<int*>(taskData->inputs[2]);
  }

  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&n, 1, MPI_INT, 0, world);

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.size() < 3) {
      return false;
    }

    double a_val = *reinterpret_cast<double*>(taskData->inputs[0]);
    double b_val = *reinterpret_cast<double*>(taskData->inputs[1]);
    if (a_val >= b_val) {
      return false;
    }

    int n_val = *reinterpret_cast<int*>(taskData->inputs[2]);
    if (n_val <= 0) {
      return false;
    }
  }

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  local_sum_ = integrate(f, a, b, n);
  MPI_Reduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_sum_;
  }

  return true;
}
