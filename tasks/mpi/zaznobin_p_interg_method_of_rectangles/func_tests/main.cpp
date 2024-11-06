#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::tuple<double, double, int> generate_random_data() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> bounds_dist(0.0, 10.0);
  std::uniform_int_distribution<> intervals_dist(100000, 2000000);

  double lower_bound = bounds_dist(gen);
  double upper_bound = lower_bound + bounds_dist(gen);
  int num_intervals = intervals_dist(gen);

  return std::make_tuple(lower_bound, upper_bound, num_intervals);
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, Test_Constant) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000;
  std::vector<double> global_sum(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return 10.0; };
  testMpiTaskParallel.get_func(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.get_func([](double x) { return 10.0; });

    ASSERT_EQ(sequential_Task.validation(), true);
    sequential_Task.pre_processing();
    sequential_Task.run();
    sequential_Task.post_processing();

    ASSERT_NEAR(global_sum[0], result_seq[0], 1e-3);
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, Test_Logarithm) {
  boost::mpi::communicator world;
  double a = 0.1;
  double b = 1.0;
  int n = 10000;
  std::vector<double> global_sum(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return std::log(x); };
  testMpiTaskParallel.get_func(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.get_func([](double x) { return std::log(x); });

    ASSERT_EQ(sequential_Task.validation(), true);
    sequential_Task.pre_processing();
    sequential_Task.run();
    sequential_Task.post_processing();

    ASSERT_NEAR(global_sum[0], result_seq[0], 1e-3);
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, Test_Gaussian) {
  boost::mpi::communicator world;
  double a = -1.0;
  double b = 1.0;
  int n = 1000;
  std::vector<double> global_sum(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return std::exp(-x * x); };
  testMpiTaskParallel.get_func(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.get_func([](double x) { return std::exp(-x * x); });

    ASSERT_EQ(sequential_Task.validation(), true);
    sequential_Task.pre_processing();
    sequential_Task.run();
    sequential_Task.post_processing();

    ASSERT_NEAR(global_sum[0], result_seq[0], 1e-2);
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, Test_Power) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000;
  std::vector<double> global_sum(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return x * x; };
  testMpiTaskParallel.get_func(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.get_func([](double x) { return x * x; });

    ASSERT_EQ(sequential_Task.validation(), true);
    sequential_Task.pre_processing();
    sequential_Task.run();
    sequential_Task.post_processing();

    ASSERT_NEAR(global_sum[0], result_seq[0], 1e-2);
  }
}
