// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaznobin_p_interg_method_of_rectangles/include/ops_mpi.hpp"

TEST(mpi_example_perf_test, test_pipeline_run) {
    boost::mpi::communicator world;
    double a = 0.0;
    double b = 1.0;
    int n = 1000000;
    double res = 0.0;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
        taskDataPar->outputs_count.emplace_back(1);
    }

    auto testMpiTaskParallel = std::make_shared<zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel>(taskDataPar);
    testMpiTaskParallel->get_func([](double x) { return x * x; });  // Используем x^2

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
        double expected_result = (b * b * b - a * a * a) / 3;  // Интеграл x^2 от a до b
        ASSERT_NEAR(res, expected_result, 1e-5);
    }
}

TEST(mpi_example_perf_test, test_task_run) {
    boost::mpi::communicator world;
    double a = 0.0;
    double b = 1.0;
    int n = 1000000;
    double res = 0.0;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
        taskDataPar->outputs_count.emplace_back(1);
    }

    auto testMpiTaskParallel = std::make_shared<zaznobin_p_interg_method_of_rectangles_mpi::TestMPITaskParallel>(taskDataPar);
    testMpiTaskParallel->get_func([](double x) { return x * x; });  // Используем x^2

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
        double expected_result = (b * b * b - a * a * a) / 3;  // Интеграл x^2 от a до b
        ASSERT_NEAR(res, expected_result, 1e-5);
    }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
