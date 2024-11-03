// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaznobin_p_interg_method_of_rectangles_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override; //Обработать данные так, как мы хотим видеть в нашей программе
  bool validation() override; //Проверка на адектватность
  bool run() override; // Тело программы
  bool post_processing() override; // Выдать результаты в удобоваримом виде для пользователя

 private:
  int input_{}, res{};
};

}  // namespace nesterov_a_test_task_seq