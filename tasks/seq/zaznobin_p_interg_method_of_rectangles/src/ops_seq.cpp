#include "seq/zaznobin_p_interg_method_of_rectangles/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

void zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::get_func(const std::function<double(double)>& func){
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
  if (n <= 0 || b <= a){
      cout << "Uncorrect start data";
      return false;
  }
  if !func{
      cout << "Func didn't get";
      return false;
  }
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1
}

double zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential::integrate(const std::function<double(double)>& f, double a, double b, int n){
  double integral = 0.;
  double h = (b-a)/n;
  for (x = a; x <= b; x+=h){
    integral+=func(x)*h;
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
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
