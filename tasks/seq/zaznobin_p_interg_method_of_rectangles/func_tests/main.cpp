#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/zaznobin_p_interg_method_of_rectangles/include/ops_seq.hpp"

TEST(zaznobin_p_interg_method_of_rectangles_seq, degree_seq) {
  double a = 0.0;
  double b = 1.0;
  int n = 1000;

  const double ans = 1 / 4;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(3);

  double res = 0.0;  // Инициализируем значение результата
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));  // Передаем указатель на результат
  taskDataSeq->outputs_count.emplace_back(1);

  zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  std::function<double(double)> f = [](double x) { return x * x * x; };
  TestTaskSequential.get_func(f);  // Или function_set(f), если это так же написано в другом коде.
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_NEAR(res, ans, 1e-3);
}

TEST(zaznobin_p_interg_method_of_rectangles_seq, sin_seq) {
  double a = 0.0;
  double b = M_PI;
  int n = 1000;

  const double ans = 2.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(3);

  double res = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  taskDataSeq->outputs_count.emplace_back(1);

  zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  std::function<double(double)> f = [](double x) { return std::sin(x); };
  TestTaskSequential.get_func(f);
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_NEAR(res, ans, 1e-3);
}

TEST(Sequential, exp_seq) {
  double a = 0.0;
  double b = 1.0;
  int n = 1000;

  // Ожидаемый результат для интеграла exp(2x) от 0 до 1
  const double ans = (std::exp(2) - 1) / 2;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(3);

  double res = 0.0;  // Инициализируем результат
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));  // Передаем указатель на результат
  taskDataSeq->outputs_count.emplace_back(1);

  zaznobin_p_interg_method_of_rectangles_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  std::function<double(double)> f = [](double x) { return std::exp(2 * x); };
  TestTaskSequential.get_func(f);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_NEAR(res, ans, 1e-3);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
