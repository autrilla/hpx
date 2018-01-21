//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_LOCAL_EDF)
#define HPX_THREADMANAGER_SCHEDULING_LOCAL_EDF

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/edf_queue.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util_fwd.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx {
namespace threads {
    namespace policies {
        template <typename Mutex = compat::mutex,
            typename PendingQueuing = lockfree_fifo,
            typename StagedQueuing = lockfree_fifo,
            typename TerminatedQueuing = lockfree_lifo>
        class HPX_EXPORT edf_scheduler : public scheduler_base
        {
        public:
            typedef std::false_type has_periodic_maintenance;

            typedef edf_queue<Mutex, PendingQueuing, StagedQueuing,
                TerminatedQueuing>
                thread_queue_type;

            struct init_parameter
            {
                init_parameter(std::size_t num_queues, char const* description)
                  : num_queues_(num_queues)
                  , description_(description)
                {
                }

                std::size_t num_queues_;
                char const* description_;
            };
            typedef init_parameter init_parameter_type;

            edf_scheduler(std::size_t num_thread)
              : scheduler_base(num_thread, "edf_scheduler")
              , queue_lock_()
              , memory_pool_(64)
            {
                queue_ = std::make_unique<thread_queue_type>(1000);
            }

            edf_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
              : scheduler_base(init.num_queues_, "edf_scheduler")
              , queue_lock_()
              , memory_pool_(64)
            {
                queue_ = std::make_unique<thread_queue_type>(1000);
            }

            virtual ~edf_scheduler() {}

            bool numa_sensitive() const
            {
                return false;
            }

            static std::string get_scheduler_name()
            {
                return "edf_scheduler";
            }

            void abort_all_suspended_threads()
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                queue_->abort_all_suspended_threads();
            }

            bool cleanup_terminated(bool delete_all = false)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->cleanup_terminated(delete_all);
            }

            void create_thread(thread_init_data& data, thread_id_type* id,
                thread_state_enum initial_state, bool run_now, error_code& ec,
                std::size_t num_thread)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                queue_->create_thread(data, id, initial_state, run_now, ec);
            }

            std::chrono::steady_clock::time_point peek_next_deadline()
            {
                return queue_->peek_next_deadline();
            }

            void peek_next_thread(threads::thread_data*& thrd)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->peek_next_thread(thrd);
            }

            virtual bool get_next_thread(std::size_t num_thread, bool running,
                std::int64_t& idle_loop_count, threads::thread_data*& thrd)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->get_next_thread(thrd);
            }

            void schedule_thread(threads::thread_data* thrd,
                std::size_t num_thread,
                thread_priority priority = thread_priority_normal)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                queue_->schedule_thread(thrd);
            }

            void schedule_thread_last(threads::thread_data* thrd,
                std::size_t num_thread,
                thread_priority priority = thread_priority_normal)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                queue_->schedule_thread(thrd, true);
            }

            bool destroy_thread(
                threads::thread_data* thrd, std::int64_t& busy_count)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->destroy_thread(thrd, busy_count);
            }

            std::int64_t get_queue_length(
                std::size_t num_thread = std::size_t(-1)) const
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->get_queue_length();
            }

            std::int64_t get_thread_count(thread_state_enum state = unknown,
                thread_priority priority = thread_priority_default,
                std::size_t num_thread = std::size_t(-1),
                bool reset = false) const
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                // Return thread count of one specific queue.
                std::int64_t count = 0;
                if (std::size_t(-1) != num_thread)
                {
                    switch (priority)
                    {
                    case thread_priority_default:
                    case thread_priority_low:
                    case thread_priority_normal:
                    case thread_priority_boost:
                    case thread_priority_high:
                    case thread_priority_high_recursive:
                        return queue_->get_thread_count(state);

                    default:
                    case thread_priority_unknown:
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "edf_scheduler::get_thread_count",
                            "unknown thread priority value "
                            "(thread_priority_unknown)");
                        return 0;
                    }
                    }
                    return 0;
                }

                // Return the cumulative count for all queues.
                switch (priority)
                {
                case thread_priority_default:
                case thread_priority_low:
                case thread_priority_normal:
                case thread_priority_boost:
                case thread_priority_high:
                case thread_priority_high_recursive:
                {
                    count = queue_->get_thread_count(state);
                    break;
                }

                default:
                case thread_priority_unknown:
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "edf_scheduler::get_thread_count",
                        "unknown thread priority value "
                        "(thread_priority_unknown)");
                    return 0;
                }
                }
                return count;
            }

            bool enumerate_threads(
                util::function_nonser<bool(thread_id_type)> const& f,
                thread_state_enum state = unknown) const
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                return queue_->enumerate_threads(f, state);
            }

            virtual bool wait_or_add_new(std::size_t num_thread, bool running,
                std::int64_t& idle_loop_count)
            {
                std::lock_guard<std::mutex> lock{queue_lock_};
                std::size_t added = 0;
                return queue_->wait_or_add_new(running, idle_loop_count, added);
            }

            // Called whenever a new OS thread is started
            void on_start_thread(std::size_t num_thread) {}

            // Called whenever an OS thread is stopped
            void on_stop_thread(std::size_t num_thread) {}

            void on_error(std::size_t num_thread, std::exception_ptr const& e)
            {
            }

        private:
            mutable std::mutex queue_lock_;
            threads::thread_pool memory_pool_;

        protected:
            std::vector<thread_data*> threads_;
            std::unique_ptr<thread_queue_type> queue_;
        };
    }
}
}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
