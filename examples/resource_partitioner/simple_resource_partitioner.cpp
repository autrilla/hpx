//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/executors.hpp>
//
#include <hpx/runtime/resource/partitioner.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool_impl.hpp>
#include <hpx/runtime/threads/executors/default_edf_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/policies/edf_scheduler.hpp>
//
#include <hpx/include/iostreams.hpp>
#include <hpx/include/runtime.hpp>
//
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
//
#include "system_characteristics.hpp"

using edf_sched = hpx::threads::policies::edf_scheduler<>;
using namespace hpx::threads::policies;

enum class Op
{
    multiply,
    multiply_sequential,
    multiply_parallel,
    multiply_parallel_noyield,
    multiply_parallel_yieldmaybe,
    multiply_parallel_yieldmaybe_synchronized,
};

struct time_info
{
    Op op;
    long long microseconds;
};

std::vector<time_info> info;

void yield_maybe_synchronized()
{
    hpx::threads::thread_data* next_thread;
    auto self = hpx::this_thread::get_id().native_handle().get();
    auto scheduler = dynamic_cast<edf_sched*>(self->get_scheduler_base());
    scheduler->peek_next_thread(next_thread);
    if (next_thread != NULL && self != NULL)
    {
        if (next_thread->get_deadline() < self->get_deadline())
        {
            hpx::this_thread::yield();
        }
    }
}

void yield_maybe()
{
    auto self = hpx::this_thread::get_id().native_handle().get();
    auto scheduler = dynamic_cast<edf_sched*>(self->get_scheduler_base());
    auto next_deadline = scheduler->peek_next_deadline();
    if (self != NULL)
    {
        if (next_deadline < self->get_deadline())
        {
            hpx::this_thread::yield();
        }
    }
}

void work_for(std::chrono::steady_clock::duration duration)
{
    const std::chrono::steady_clock::duration MAX_SLEEP_DURATION =
        std::chrono::microseconds(100);
    auto sleep_duration = MAX_SLEEP_DURATION;
    if (duration < sleep_duration)
    {
        sleep_duration = duration;
    }
    if (sleep_duration.count() == 0)
    {
        return;
    }
    auto sleep_count = duration / sleep_duration;
    for (auto i = 0; i < sleep_count; i++)
    {
        std::this_thread::sleep_for(sleep_duration);
        yield_maybe();
    }
}

void work_for_noyield(std::chrono::steady_clock::duration duration)
{
    std::this_thread::sleep_for(duration);
}

void work_for_print(std::chrono::steady_clock::duration duration)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    work_for(duration);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Worked for " << d << " microseconds, was told to work for "
              << std::chrono::duration_cast<std::chrono::microseconds>(duration)
                     .count()
              << " microseconds." << std::endl;
}

void no_work_task_overhead()
{
    for (auto i = 0; i < 100; i++)
    {
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
        hpx::async([]() { work_for(std::chrono::microseconds(0)); }).get();
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count();
        std::cout << "Running an empty task took " << d << " microseconds."
                  << std::endl;
    }
}

void work_for_overhead()
{
    hpx::async([]() {
        work_for_print(std::chrono::seconds(1));
        work_for_print(std::chrono::milliseconds(100));
        work_for_print(std::chrono::milliseconds(10));
        work_for_print(std::chrono::milliseconds(1));
        work_for_print(std::chrono::microseconds(100));
        work_for_print(std::chrono::microseconds(10));
    })
        .get();
}

void yield_test()
{
    hpx::async([]() {
        while (true)
        {
            std::chrono::steady_clock::time_point t1 =
                std::chrono::steady_clock::now();
            yield_maybe();
            std::chrono::steady_clock::time_point t2 =
                std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                    .count();
            std::cout << "yielding took " << duration << " microseconds"
                      << std::endl;
        }
    })
        .get();
}

std::vector<std::vector<int>> multiply_matrices(std::vector<std::vector<int>> a,
    std::vector<std::vector<int>>
        b)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    for (size_t i = 0; i < a_rows; i++)
    {
        result[i] = std::vector<int>(b_cols);
        for (size_t j = 0; j < b_cols; j++)
        {
            auto v = 0;
            for (size_t k = 0; k < a_cols; k++)
            {
                v += a[i][k] * b[k][j];
                result[i][j] = v;
            }
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(time_info{Op::multiply, duration});
    return result;
}

std::vector<std::vector<int>>
multiply_matrices_sequential(std::vector<std::vector<int>> a,
    std::vector<std::vector<int>>
        b)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    hpx::async([a_rows, a_cols, b_cols, &a, &b, &result]() {
        for (size_t i = 0; i < a_rows; i++)
        {
            result[i] = std::vector<int>(b_cols);
            for (size_t j = 0; j < b_cols; j++)
            {
                auto v = 0;
                for (size_t k = 0; k < a_cols; k++)
                {
                    v += a[i][k] * b[k][j];
                    result[i][j] = v;
                }
            }
        }
    })
        .get();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(time_info{Op::multiply_sequential, duration});
    return result;
}

std::vector<std::vector<int>>
multiply_matrices_parallel(std::vector<std::vector<int>> a,
    std::vector<std::vector<int>>
        b)
{
    hpx::threads::executors::default_edf_executor executor2(
        std::chrono::steady_clock::now());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    std::vector<hpx::future<int>> futures(a_rows);
    for (size_t i = 0; i < a_rows; i++)
    {
        futures[i] =
            hpx::async(executor2, [a_cols, b_cols, i, &a, &b, &result]() {
                result[i] = std::vector<int>(b_cols);
                for (size_t j = 0; j < b_cols; j++)
                {
                    auto v = 0;
                    for (size_t k = 0; k < a_cols; k++)
                    {
                        v += a[i][k] * b[k][j];
                    }
                    hpx::this_thread::yield();
                    result[i][j] = v;
                }
                return 0;
            });
    }
    hpx::wait_all(futures);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(time_info{Op::multiply_parallel, duration});
    return result;
}

std::vector<std::vector<int>>
multiply_matrices_parallel_noyield(std::vector<std::vector<int>> a,
    std::vector<std::vector<int>>
        b)
{
    hpx::threads::executors::default_edf_executor executor2(
        std::chrono::steady_clock::now());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    std::vector<hpx::future<int>> futures(a_rows);
    for (size_t i = 0; i < a_rows; i++)
    {
        futures[i] =
            hpx::async(executor2, [a_cols, b_cols, i, &a, &b, &result]() {
                result[i] = std::vector<int>(b_cols);
                for (size_t j = 0; j < b_cols; j++)
                {
                    auto v = 0;
                    for (size_t k = 0; k < a_cols; k++)
                    {
                        v += a[i][k] * b[k][j];
                    }
                    result[i][j] = v;
                }
                return 0;
            });
    }
    hpx::wait_all(futures);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(time_info{Op::multiply_parallel_noyield, duration});
    return result;
}

std::vector<std::vector<int>>
multiply_matrices_parallel_yieldmaybe(std::vector<std::vector<int>> a,
    std::vector<std::vector<int>>
        b)
{
    hpx::threads::executors::default_edf_executor executor2(
        std::chrono::steady_clock::now());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    std::vector<hpx::future<int>> futures(a_rows);
    for (size_t i = 0; i < a_rows; i++)
    {
        futures[i] =
            hpx::async(executor2, [a_cols, b_cols, i, &a, &b, &result]() {
                result[i] = std::vector<int>(b_cols);
                for (size_t j = 0; j < b_cols; j++)
                {
                    auto v = 0;
                    for (size_t k = 0; k < a_cols; k++)
                    {
                        v += a[i][k] * b[k][j];
                    }
                    yield_maybe();
                    result[i][j] = v;
                }
                return 0;
            });
    }
    hpx::wait_all(futures);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(time_info{Op::multiply_parallel_yieldmaybe, duration});
    return result;
}

std::vector<std::vector<int>>
multiply_matrices_parallel_yieldmaybe_synchronized(
    std::vector<std::vector<int>> a, std::vector<std::vector<int>> b)
{
    hpx::threads::executors::default_edf_executor executor2(
        std::chrono::steady_clock::now());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto const a_rows = a.size();
    auto const a_cols = a[0].size();
    auto const b_cols = b[0].size();
    std::vector<std::vector<int>> result(a_rows);
    std::vector<hpx::future<int>> futures(a_rows);
    for (size_t i = 0; i < a_rows; i++)
    {
        futures[i] =
            hpx::async(executor2, [a_cols, b_cols, i, &a, &b, &result]() {
                result[i] = std::vector<int>(b_cols);
                for (size_t j = 0; j < b_cols; j++)
                {
                    auto v = 0;
                    for (size_t k = 0; k < a_cols; k++)
                    {
                        v += a[i][k] * b[k][j];
                    }
                    yield_maybe_synchronized();
                    result[i][j] = v;
                }
                return 0;
            });
    }
    hpx::wait_all(futures);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    info.push_back(
        time_info{Op::multiply_parallel_yieldmaybe_synchronized, duration});
    return result;
}

void print_matrix(std::vector<std::vector<int>> m)
{
    for (size_t i = 0; i < m.size(); i++)
    {
        for (size_t j = 0; j < m[i].size(); j++)
        {
            std::cout << m[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

enum class task_timing_event
{
    activation,
    work_started,
    work_done,
};

struct task_timing
{
    task_timing_event event;
    std::chrono::steady_clock::time_point time;
    size_t activation;
};

class task
{
public:
    task(size_t task_id,
        std::chrono::steady_clock::duration activation_interval,
        std::chrono::steady_clock::duration relative_deadline,
        std::chrono::steady_clock::duration work_duration)
      : task_timings_()
      , task_id_(task_id)
      , activation_interval_(activation_interval)
      , relative_deadline_(relative_deadline)
      , work_duration_(work_duration)
      , timer_([this]() { return this->do_work(); }, activation_interval)
    {
        task_timings_.reserve(
            4 * (std::chrono::seconds(10) / activation_interval_));
    }

    task(const task& t)
      : task_timings_()
      , task_id_(t.task_id_)
      , activation_interval_(t.activation_interval_)
      , relative_deadline_(t.relative_deadline_)
      , work_duration_(t.work_duration_)
      , timer_([this]() { return this->do_work(); }, t.activation_interval_)
    {
        task_timings_.reserve(
            4 * (std::chrono::seconds(10) / activation_interval_));
    }

    void start()
    {
        current_activation_ = 0;
        last_finished_activation_ = 0;
        timer_.start();
    }

    void stop()
    {
        timer_.stop();
    }

    std::vector<task_timing> task_timings_;
    size_t task_id_;
    std::chrono::steady_clock::duration activation_interval_;
    std::chrono::steady_clock::duration relative_deadline_;
    std::chrono::steady_clock::duration work_duration_;

private:
    bool do_work()
    {
        auto activation = this->current_activation_;
        if (last_finished_activation_ + 1 < activation)
        {
            std::cout << "Task " << task_id_
                      << " is overwhelmed. (last finished "
                      << last_finished_activation_ << " current " << activation
                      << ")" << std::endl;
        }
        this->current_activation_++;
        auto now = std::chrono::steady_clock::now();
        task_timings_.push_back(
            task_timing{task_timing_event::activation, now, activation});
        auto deadline = now + relative_deadline_;
        hpx::threads::executors::default_edf_executor executor(deadline);
        hpx::async(executor, [this, activation]() {
            task_timings_.push_back(task_timing{task_timing_event::work_started,
                std::chrono::steady_clock::now(), activation});
            work_for_noyield(work_duration_);
            task_timings_.push_back(task_timing{task_timing_event::work_done,
                std::chrono::steady_clock::now(), activation});
            last_finished_activation_ = activation;
        });
        return true;
    }
    hpx::util::interval_timer timer_;
    size_t current_activation_;
    size_t last_finished_activation_;
};

void matrix_tests()
{
    const size_t size = 500;
    std::vector<std::vector<int>> a(size);
    for (size_t i = 0; i < a.size(); i++)
    {
        auto tmp = std::vector<int>(size);
        std::generate(tmp.begin(), tmp.end(), std::rand);
        a[i] = tmp;
    }
    std::vector<std::vector<int>> b(size);
    for (size_t i = 0; i < a.size(); i++)
    {
        auto tmp = std::vector<int>(size);
        std::generate(tmp.begin(), tmp.end(), std::rand);
        b[i] = tmp;
    }
    auto const ITERS = 100;
    info.reserve(6 * ITERS);
    for (auto i = 0; i < ITERS; i++)
    {
        multiply_matrices_sequential(a, b);
        multiply_matrices(a, b);
        multiply_matrices_parallel_noyield(a, b);
        multiply_matrices_parallel_yieldmaybe_synchronized(a, b);
        multiply_matrices_parallel_yieldmaybe(a, b);
        multiply_matrices_parallel(a, b);
    }

    for (size_t i = 0; i < info.size(); i++)
    {
        auto t = info[i];
        std::string name;
        switch (t.op)
        {
        case Op::multiply:
            name = "multiply";
            break;
        case Op::multiply_sequential:
            name = "multiply_sequential";
            break;
        case Op::multiply_parallel:
            name = "multiply_parallel";
            break;
        case Op::multiply_parallel_noyield:
            name = "multiply_parallel_noyield";
            break;
        case Op::multiply_parallel_yieldmaybe:
            name = "multiply_parallel_yieldmaybe";
            break;
        case Op::multiply_parallel_yieldmaybe_synchronized:
            name = "multiply_parallel_yieldmaybe_synchronized";
            break;
        }
        std::cout << name << " took " << t.microseconds << " microseconds."
                  << std::endl;
    }
}

void tasks_tests()
{
    std::ifstream infile("tasks.txt");
    std::vector<task> tasks;
    size_t task_id = 0;
    int period, work_time;
    while (infile >> period >> work_time)
    {
        tasks.push_back(task(task_id++, std::chrono::microseconds(period),
            std::chrono::microseconds(period),
            std::chrono::microseconds(work_time)));
    }
    float total_utilization = 0;
    for (size_t i = 0; i < tasks.size(); i++)
    {
        total_utilization += (float) tasks[i].work_duration_.count() /
            (float) tasks[i].activation_interval_.count();
        tasks[i].start();
    }
    std::cout << "Total utilization: " << total_utilization << std::endl;
    std::cout << "Started all tasks" << std::endl;
    hpx::this_thread::sleep_for(std::chrono::seconds(10));
    for (size_t i = 0; i < tasks.size(); i++)
    {
        tasks[i].stop();
    }
    std::cout << "Stopped all tasks" << std::endl;
    hpx::this_thread::sleep_for(std::chrono::seconds(2));
    std::ofstream out;
    out.open("/tmp/out.txt");
    for (size_t i = 0; i < tasks.size(); i++)
    {
        out << "task_details " << i << " period "
            << tasks[i].activation_interval_.count() << " deadline "
            << tasks[i].relative_deadline_.count() << " work_duration "
            << tasks[i].work_duration_.count() << std::endl;
        for (size_t j = 0; j < tasks[i].task_timings_.size(); j++)
        {
            auto timing = tasks[i].task_timings_[j];
            std::string name;
            switch (timing.event)
            {
            case task_timing_event::activation:
                name = "activation";
                break;
            case task_timing_event::work_started:
                name = "work_started";
                break;
            case task_timing_event::work_done:
                name = "work_done";
                break;
            default:
                std::cerr << "Received timing data with no event" << std::endl;
                exit(1);
                break;
            }
            out << "task " << i << " activation " << timing.activation << " "
                << name << " " << timing.time.time_since_epoch().count()
                << std::endl;
        }
    }
    out.close();
    std::cout << "Wrote runtime information to disk." << std::endl;
}

void test_pools()
{
    auto t1 = std::chrono::steady_clock::now();
    auto executor1 = hpx::threads::executors::pool_executor("1", t1);
    auto executor2 = hpx::threads::executors::pool_executor("2", t1);
    for (auto i = 0; i < 10; i++)
    {
        hpx::async(executor1,
            []() {
                hpx::cout << "executor1 running on scheduler "
                          << hpx::this_thread::get_id()
                                 .native_handle()
                                 .get()
                                 ->get_scheduler_base()
                          << hpx::endl;
            })
            .get();
        hpx::async(executor2,
            []() {
                hpx::cout << "executor2 running on scheduler "
                          << hpx::this_thread::get_id()
                                 .native_handle()
                                 .get()
                                 ->get_scheduler_base()
                          << hpx::endl;
            })
            .get();
    }
}

// this is called on an hpx thread after the runtime starts up
int hpx_main(boost::program_options::variables_map& vm)
{
    test_pools();
    no_work_task_overhead();
    work_for_overhead();
    tasks_tests();

    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS
// thread
// the user must call hpx::init to start the hpx runtime which will execute
// hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    boost::program_options::options_description desc_cmdline("Test options");
    hpx::resource::partitioner rp(desc_cmdline, argc, argv);
    rp.create_thread_pool("default",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
            -> std::unique_ptr<hpx::threads::detail::thread_pool_base> {
            auto s = new edf_sched(num_threads);
            std::unique_ptr<edf_sched> scheduler(s);

            auto mode = scheduler_mode(scheduler_mode::do_background_work |
                scheduler_mode::delay_exit);

            std::unique_ptr<hpx::threads::detail::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<edf_sched>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset));
            return pool;
        });

    rp.create_thread_pool("1",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
            -> std::unique_ptr<hpx::threads::detail::thread_pool_base> {
            auto s = new edf_sched(1);
            std::unique_ptr<edf_sched> scheduler(s);

            auto mode = scheduler_mode(scheduler_mode::do_background_work |
                scheduler_mode::delay_exit);

            auto thread_pool =
                new hpx::threads::detail::scheduled_thread_pool<edf_sched>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset);

            std::unique_ptr<hpx::threads::detail::thread_pool_base> pool(
                thread_pool);
            return pool;
        });

    rp.create_thread_pool("2",
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
            -> std::unique_ptr<hpx::threads::detail::thread_pool_base> {
            auto s = new edf_sched(1);
            std::unique_ptr<edf_sched> scheduler(s);

            auto mode = scheduler_mode(scheduler_mode::do_background_work |
                scheduler_mode::delay_exit);

            std::unique_ptr<hpx::threads::detail::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<edf_sched>(
                    std::move(scheduler), notifier, pool_index, pool_name, mode,
                    thread_offset));
            return pool;
        });

    int pool_1_count = 0;
    int pool_2_count = 0;
    for (const hpx::resource::numa_domain& d : rp.numa_domains())
    {
        for (const hpx::resource::core& c : d.cores())
        {
            for (const hpx::resource::pu& p : c.pus())
            {
                if (pool_1_count < 1)
                {
                    std::cout << "Added pu to pool 1\n";
                    rp.add_resource(p, "1");
                    pool_1_count++;
                }
                else if (pool_2_count < 1)
                {
                    std::cout << "Added pu to pool 2\n";
                    rp.add_resource(p, "2");
                    pool_2_count++;
                }
            }
        }
    }

    return hpx::init();
}
