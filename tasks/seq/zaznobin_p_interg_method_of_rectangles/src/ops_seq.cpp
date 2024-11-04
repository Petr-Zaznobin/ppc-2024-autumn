#include "seq/zaznobin_p_interg_method_of_rectangles/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

void zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::get_func(const std::function<double(double)>& f) {
  func = f;
}

bool zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  // Инициализация значений a, b и n из taskData
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  n = *reinterpret_cast<int*>(taskData->inputs[2]);
  res = 0.0;  // Инициализация результата
  return true;
}

bool zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (!func) {
    std::cout << "Func didn't get";
    return false;
  }
  if (taskData->inputs.size() < 3) {
    std::cout << "Not enough input data";
    return false;
  }
  // Check count elements of output
  return true;
}

double zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::integrate(const std::function<double(double)>& f,
                                                                                 double a, double b, int n) {
  double integral = 0.0;
  double h = (b - a) / n;
  
  for (int i = 0; i < n; ++i) {
    double x = a + i * h;  // или lower_bound, если нужно
    integral += f(x) * h;
  }
  
  return integral;
}

bool zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::run() {
  internal_order_test();
  res = integrate(func, a, b, n);
  return true;
}

bool zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
