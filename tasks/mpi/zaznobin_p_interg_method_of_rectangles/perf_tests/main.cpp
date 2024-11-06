#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

std::tuple<double, double, int> generate_random_data() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> bounds_dist(0.0, 10.0);
  std::uniform_int_distribution<> intervals_dist(100000, 2000000);

  double a = bounds_dist(gen);
  double b = a + bounds_dist(gen);
  int n = intervals_dist(gen);

  return std::make_tuple(a, b, n);
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->get_func([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_result = (a * a * a - b * b * b) / 3;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}

TEST(zaznobin_p_interg_method_of_rectangles_mpi, test_task_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->get_func([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_result = (a * a * a - b * b * b) / 3;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}
