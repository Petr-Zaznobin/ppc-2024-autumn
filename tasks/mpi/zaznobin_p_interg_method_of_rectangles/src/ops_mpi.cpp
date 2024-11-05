// Copyright 2023 Nesterov Alexander
#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::get_func(
    const std::function<double(double)>& f) {
  func = f;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  // Инициализация значений a, b и n из taskData
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  n = *reinterpret_cast<int*>(taskData->inputs[2]);
  res = 0.0;  // Инициализация результата
  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (!func) {
    std::cout << "Function not set" << std::endl;
    return false;
  }
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

double zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::integrate(
    const std::function<double(double)>& f, double a, double b, int n) {
  double integral = 0.0;
  double h = (b - a) / n;
  for (int i = 0; i < n; ++i) {
    double x = a + i * h;
    integral += f(x) * h;
  }
  return integral;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = integrate(func, a, b, n);
  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
//--------------------------------------------------------------
void zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::get_func(const std::function<double(double)>& f) {
  func = f;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // На процессе с rank 0 инициализируем значения
  if (world.rank() == 0) {
    auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);

    a = *tmp_ptr_a;  // Инициализация a
    b = *tmp_ptr_b;  // Инициализация b
    n = *tmp_ptr_n;  // Инициализация n
  }

  // Распространяем значения a, b и n среди всех процессов
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&n, 1, MPI_INT, 0, world);

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Проверка, что достаточно входных данных
    if (taskData->inputs.size() < 3) {
      std::cout << "not enough input data" << std::endl;
      return false;
    }

    // Проверка корректности границ интегрирования
    double a_val = *reinterpret_cast<double*>(taskData->inputs[0]);
    double b_val = *reinterpret_cast<double*>(taskData->inputs[1]);
    if (a_val >= b_val) {
      std::cout << "a>=b" << std::endl;
      return false;
    }

    // Проверка, что количество интервалов положительное
    int n_val = *reinterpret_cast<int*>(taskData->inputs[2]);
    if (n_val <= 0) {
      std::cout << "n <= 0." << std::endl;
      return false;
    }

    // Проверка, что функция для интегрирования задана
    if (!func) {
      std::cout << "integration function is not set." << std::endl;
      return false;
    }
  }

  // Обеспечиваем синхронизацию всех процессов перед выполнением расчета
  MPI_Barrier(world);

  return true;
}

double zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::integrate(
    const std::function<double(double)>& f, double a, double b, int n) {
  // Определяем количество процессов
  int num_procs = world.size();
  int rank = world.rank();

  double width = (b - a) / n;  // Шаг по оси x для метода прямоугольников

  // Минимальное количество интервалов для каждого процесса
  int local_num_intervals = n / num_procs;
  int remainder = n % num_procs;

  // Процессы с rank < remainder будут обрабатывать на один интервал больше
  if (rank < remainder) {
    local_num_intervals += 1;
  }
  // Вычисляем начальную точку интегрирования для каждого процесса
  double local_start = a + rank * (n / num_procs) * width;
  if (rank < remainder) {
    local_start += rank * width;  // Добавляем смещение для процессов с дополнительным интервалом
  } else {
    local_start += remainder * width;  // Смещение для остальных процессов
  }
  // Локальное вычисление интеграла по методу прямоугольников
  double local_sum = 0.0;
  for (int i = 0; i < local_num_intervals; ++i) {
    double x = local_start + i * width;
    local_sum += f(x) * width;
  }

  return local_sum;
}


bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  // Локальное вычисление интеграла
  local_sum_ = integrate(func, a, b, n);

  // Сбор результатов
  MPI_Reduce(&local_sum_, &res, 1, MPI_DOUBLE, MPI_SUM, 0, world);

  return true;
}

bool zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}
