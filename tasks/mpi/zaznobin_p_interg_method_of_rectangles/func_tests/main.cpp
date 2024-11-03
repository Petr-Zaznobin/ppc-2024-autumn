// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

TEST(zaznobin_p_interg_method_of_rectangles_mpi, Sin_mpi) {
  boost::mpi::communicator world;

  double a = 0.0;
  double b = M_PI;
  int n = 1000;
  double global_result = 0.0;
  double sequential_result = 0.0;

  // Создаем объект TaskData для параллельной задачи
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Параллельная задача
  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.get_func([](double x) { return std::sin(x); });

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем объект TaskData для последовательной задачи
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));
    taskDataSeq->outputs_count.emplace_back(1);

    // Последовательная задача
    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.get_func([](double x) { return std::sin(x); });

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Сравнение результатов параллельного и последовательного вычислений
    ASSERT_NEAR(global_result, sequential_result, 1e-5);
    ASSERT_NEAR(global_result, 2.0, 1e-5);  // Сравнение с точным значением интеграла
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, exp_mpi) {
  boost::mpi::communicator world;

  double a = 0.0;
  double b = 1.0;
  int n = 1000;
  double global_result = 0.0;
  double sequential_result = 0.0;

  // Создаем объект TaskData для параллельной задачи
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Параллельная задача
  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.get_func([](double x) { return std::exp(2 * x); });

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем объект TaskData для последовательной задачи
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));
    taskDataSeq->outputs_count.emplace_back(1);

    // Последовательная задача
    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.get_func([](double x) { return std::exp(2 * x); });

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Сравнение результатов параллельного и последовательного вычислений
    ASSERT_NEAR(global_result, sequential_result, 1e-5);
    ASSERT_NEAR(global_result, (std::exp(2 * b) - std::exp(2 * a)) / 2.0,
                1e-5);  // Сравнение с точным значением интеграла
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, degree_mpi) {
  boost::mpi::communicator world;

  double a = 0.0;
  double b = 3.0;
  int n = 1000;
  double global_result = 0.0;
  double sequential_result = 0.0;

  // Создаем объект TaskData для параллельной задачи
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Параллельная задача
  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel parallelTask(taskDataPar);
  parallelTask.get_func([](double x) { return x * x * x; });

  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем объект TaskData для последовательной задачи
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&sequential_result));
    taskDataSeq->outputs_count.emplace_back(1);

    // Последовательная задача
    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequentialTask(taskDataSeq);
    sequentialTask.get_func([](double x) { return x * x * x; });

    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Сравнение результатов параллельного и последовательного вычислений
    ASSERT_NEAR(global_result, sequential_result, 1e-5);
    ASSERT_NEAR(global_result, 60.75, 1e-5);  // Сравнение с точным значением интеграла
  }
}

