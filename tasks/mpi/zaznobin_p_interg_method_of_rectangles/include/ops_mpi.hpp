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

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
    public:
        TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)){}
        bool pre_processing() override; //Обработать данные так, как мы хотим видеть в нашей программе
        bool validation() override; //Проверка на адектватность
        bool run() override; // Тело программы
        bool post_processing() override; // Выдать результаты в удобоваримом виде для пользователя
        void get_func(const std::function<double(double)>& func); // функция для интегрирования
    private:
        double a={};
        double b={};
        double n={};
        //std::vector<double> input_;
        std::function<double(double)> func;
        double res;
        double integrate(const std::function<double(double)>& f, double a, double b, int n);
};

class TestMPITaskParallel : public ppc::core::Task {
    public:
        explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
          : Task(std::move(taskData_)), ops(std::move(ops_)) {}
        bool pre_processing() override;
        bool validation() override;
        bool run() override;
        bool post_processing() override;
        void get_func(const std::function<double(double)>& func);
    private:
        double a={};
        double b={};
        double n={};
        std::function<double(double)> func;
        double local_sum_{};
        double gloval_res;
        double integrate(const std::function<double(double)>& f, double a, double b, int n);
        //std::vector<int> input_, local_input_;
        boost::mpi::communicator world;
};

}  // namespace nesterov_a_test_task_mpi
