#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaznobin_p_interg_method_of_rectangles_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void get_func(const std::function<double(double)>& func);

 private:
  double a{};
  double b{};
  int n{};
  std::function<double(double)> f;
  std::vector<double> input_;
  std::vector<double> results_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void get_func(const std::function<double(double)>& func);

 private:
  double integrate(const std::function<double(double)>& f_, double a_, double b_, int n_);
  double a{};
  double b{};
  double local_sum_{};
  double global_sum_{};
  int n{};
  std::function<double(double)> f;
  std::vector<double> input_;
  std::vector<double> results_;
  boost::mpi::communicator world;
};
}  // namespace zaznobin_p_interg_method_of_rectangles_mpi
