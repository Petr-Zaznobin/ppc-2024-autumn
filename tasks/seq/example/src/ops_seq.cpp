#include "seq/example/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

void zaznobin_func(std::function<double()> f){
    func = f
}

bool nesterov_a_test_task_seq::TestTaskSequential::pre_processing() {
    internal_order_test();
    // Init value for input and output
    a = reinterpret_cast<double*>(taskData->inputs[0]);
    b = reinterpret_cast<double*>(taskData->inputs[1]);
    res = 0;
    return true;
}

bool nesterov_a_test_task_seq::TestTaskSequential::validation() {
    internal_order_test();

    // Check count elements of output
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool nesterov_a_test_task_seq::TestTaskSequential::run() {
    internal_order_test();
    return true;
}

bool nesterov_a_test_task_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
