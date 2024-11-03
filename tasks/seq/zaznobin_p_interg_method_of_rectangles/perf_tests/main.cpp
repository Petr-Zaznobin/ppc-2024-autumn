// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/example/include/ops_seq.hpp"

TEST(zaznobin_p_interg_method_of_rectangles_seq, test_pipeline_run) {
    double a = 0.0;
    double b = 3.0;
    int n = 1000000;
    double expected_result = 9.0;  // Ожидаемое значение интеграла x^2 от 0 до 3

    // Настройка входных и выходных данных
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(3);

    double res = 0.0;
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Создание задачи и установка функции интегрирования
    auto testTaskSequential = std::make_shared<zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential>(taskDataSeq);
    std::function<double(double)> func = [](double x) { return x * x; };
    testTaskSequential->get_func(func);

    // Настройка атрибутов производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 1000;

    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);

    // Запуск задачи с использованием pipeline_run
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    // Печать статистики производительности
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результата
    EXPECT_NEAR(res, expected_result, 0.0001);
}

TEST(zaznobin_p_interg_method_of_rectangles_seq, test_task_run) {
    double a = 0.0;
    double b = 3.0;
    int n = 1000000;
    double expected_result = 9.0;  // Ожидаемый результат интеграла x^2 от 0 до 3

    // Настройка TaskData с входными и выходными значениями
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(3);

    double result = 0.0;  // Значение для результата интегрирования
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    taskDataSeq->outputs_count.emplace_back(1);

    // Создание задачи и установка функции
    auto testTaskSequential = std::make_shared<zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential>(taskDataSeq);
    testTaskSequential->get_func([](double x) { return x * x; });

    // Настройка атрибутов производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 1000;

    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание анализатора производительности и запуск теста
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults);

    // Вывод статистики производительности
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка полученного результата с ожидаемым
    EXPECT_NEAR(result, expected_result, 0.0001);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
