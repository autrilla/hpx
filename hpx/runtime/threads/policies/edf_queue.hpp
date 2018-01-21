//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_EDF_QUEUE)
#define HPX_THREADMANAGER_EDF_QUEUE

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/queue_helpers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/spinlock.hpp>

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
#   include <hpx/util/tick_counter.hpp>
#endif

#include <boost/lexical_cast.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    template <typename Mutex = hpx::util::spinlock,
        typename PendingQueuing = lockfree_lifo,
        typename StagedQueuing = lockfree_lifo,
        typename TerminatedQueuing = lockfree_fifo>
    class edf_queue
    {
    private:
        // we use a simple mutex to protect the data members for now
        typedef Mutex mutex_type;

        // don't steal if less than this amount of tasks are left
        int const min_tasks_to_steal_pending;
        int const min_tasks_to_steal_staged;

        // create at least this amount of threads from tasks
        int const min_add_new_count;

        // create not more than this amount of threads from tasks
        int const max_add_new_count;

        // number of terminated threads to discard
        int const max_delete_count;

        // number of terminated threads to collect before cleaning them up
        int const max_terminated_threads;

        // this is the type of a map holding all threads (except depleted ones)
        typedef std::unordered_set<thread_id_type> thread_map_type;

        typedef util::tuple<thread_init_data, thread_state_enum> task_description;
        typedef thread_data thread_description;
        
        static bool work_items_ordering(thread_description* left, thread_description* right) {
            return left->get_deadline() > right->get_deadline();
        };

        typedef std::priority_queue<
            thread_description*,
            std::vector<thread_description*>,
            decltype(&work_items_ordering)
        > work_items_type;

        typedef typename StagedQueuing::template
            apply<task_description*>::type task_items_type;

        typedef typename TerminatedQueuing::template
            apply<thread_data*>::type terminated_items_type;

    protected:
        template <typename Lock>
        void create_thread_object(threads::thread_id_type& thrd,
            threads::thread_init_data& data, thread_state_enum state, Lock& lk)
        {
            HPX_ASSERT(lk.owns_lock());
            HPX_ASSERT(data.stacksize != 0);

            std::ptrdiff_t stacksize = data.stacksize;

            std::list<thread_id_type>* heap = nullptr;

            if (stacksize == get_stack_size(thread_stacksize_small))
            {
                heap = &thread_heap_small_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_medium))
            {
                heap = &thread_heap_medium_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_large))
            {
                heap = &thread_heap_large_;
            }
            else if (stacksize == get_stack_size(thread_stacksize_huge))
            {
                heap = &thread_heap_huge_;
            }
            else {
                switch(stacksize) {
                case thread_stacksize_small:
                    heap = &thread_heap_small_;
                    break;

                case thread_stacksize_medium:
                    heap = &thread_heap_medium_;
                    break;

                case thread_stacksize_large:
                    heap = &thread_heap_large_;
                    break;

                case thread_stacksize_huge:
                    heap = &thread_heap_huge_;
                    break;

                default:
                    break;
                }
            }
            HPX_ASSERT(heap);

            if (state == pending_do_not_schedule || state == pending_boost)
            {
                state = pending;
            }

            // Check for an unused thread object.
            if (!heap->empty())
            {
                // Take ownership of the thread object and rebind it.
                thrd = heap->front();
                heap->pop_front();
                thrd->rebind(data, state);
            }

            else
            {
                hpx::util::unlock_guard<Lock> ull(lk);

                // Allocate a new thread object.
                thrd = threads::thread_data::create(data, memory_pool_, state);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // add new threads if there is some amount of work available
        std::size_t add_new(std::int64_t add_count, edf_queue* addfrom,
            std::unique_lock<mutex_type> &lk, bool steal = false)
        {
            HPX_ASSERT(lk.owns_lock());

            if (HPX_UNLIKELY(0 == add_count))
                return 0;

            std::size_t added = 0;
            task_description* task = nullptr;
            while (add_count-- && addfrom->new_tasks_.pop(task, steal))
            {
                --addfrom->new_tasks_count_;

                // measure thread creation time
                util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

                // create the new thread
                threads::thread_init_data& data = util::get<0>(*task);
                thread_state_enum state = util::get<1>(*task);
                threads::thread_id_type thrd;

                create_thread_object(thrd, data, state, lk);

                delete task;

                // add the new entry to the map of all threads
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(thrd);

                if (HPX_UNLIKELY(!p.second)) {
                    lk.unlock();
                    HPX_THROW_EXCEPTION(hpx::out_of_memory,
                        "threadmanager::add_new",
                        "Couldn't add new thread to the thread map");
                    return 0;
                }
                ++thread_map_count_;

                // only insert the thread into the work-items queue if it is in
                // pending state
                if (state == pending) {
                    // pushing the new thread into the pending queue of the
                    // specified edf_queue
                    ++added;
                    schedule_thread(thrd.get());
                }

                // this thread has to be in the map now
                HPX_ASSERT(thread_map_.find(thrd.get()) != thread_map_.end());
                HPX_ASSERT(thrd->get_pool() == &memory_pool_);
            }

            if (added) {
                LTM_(debug) << "add_new: added " << added << " tasks to queues"; //-V128
            }
            return added;
        }

        ///////////////////////////////////////////////////////////////////////
        bool add_new_always(std::size_t& added, edf_queue* addfrom,
            std::unique_lock<mutex_type> &lk, bool steal = false)
        {
            HPX_ASSERT(lk.owns_lock());

            // create new threads from pending tasks (if appropriate)
            std::int64_t add_count = -1;            // default is no constraint

            // if we are desperate (no work in the queues), add some even if the
            // map holds more than max_count
            if (HPX_LIKELY(max_count_)) {
                std::size_t count = thread_map_.size();
                if (max_count_ >= count + min_add_new_count) { //-V104
                    HPX_ASSERT(max_count_ - count <
                        static_cast<std::size_t>(
                            (std::numeric_limits<std::int64_t>::max)()
                        ));
                    add_count = static_cast<std::int64_t>(max_count_ - count);
                    if (add_count < min_add_new_count)
                        add_count = min_add_new_count;
                    if (add_count > max_add_new_count)
                        add_count = max_add_new_count;
                }
                else if (work_items_.empty()) {
                    add_count = min_add_new_count;    // add this number of threads
                    max_count_ += min_add_new_count;  // increase max_count //-V101
                }
                else {
                    return false;
                }
            }

            std::size_t addednew = add_new(add_count, addfrom, lk, steal);
            added += addednew;
            return addednew != 0;
        }

        void recycle_thread(thread_id_type thrd)
        {
            std::ptrdiff_t stacksize = thrd->get_stack_size();

            if (stacksize == get_stack_size(thread_stacksize_small))
            {
                thread_heap_small_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_medium))
            {
                thread_heap_medium_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_large))
            {
                thread_heap_large_.push_front(thrd);
            }
            else if (stacksize == get_stack_size(thread_stacksize_huge))
            {
                thread_heap_huge_.push_front(thrd);
            }
            else
            {
                switch(stacksize) {
                case thread_stacksize_small:
                    thread_heap_small_.push_front(thrd);
                    break;

                case thread_stacksize_medium:
                    thread_heap_medium_.push_front(thrd);
                    break;

                case thread_stacksize_large:
                    thread_heap_large_.push_front(thrd);
                    break;

                case thread_stacksize_huge:
                    thread_heap_huge_.push_front(thrd);
                    break;

                default:
                    HPX_ASSERT(false);
                    break;
                }
            }
        }

    public:
        /// This function makes sure all threads which are marked for deletion
        /// (state is terminated) are properly destroyed.
        ///
        /// This returns 'true' if there are no more terminated threads waiting
        /// to be deleted.
        bool cleanup_terminated_locked_helper(bool delete_all = false)
        {

            if (terminated_items_count_ == 0 && thread_map_.empty())
                return true;

            if (delete_all) {
                // delete all threads
                thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    --terminated_items_count_;

                    // this thread has to be in this map
                    HPX_ASSERT(thread_map_.find(todelete) != thread_map_.end());

                    bool deleted = thread_map_.erase(todelete) != 0;
                    HPX_ASSERT(deleted);
                    if (deleted) {
                        --thread_map_count_;
                        HPX_ASSERT(thread_map_count_ >= 0);
                    }
                }
            }
            else {
                // delete only this many threads
                std::int64_t delete_count =
                    (std::max)(
                        static_cast<std::int64_t>(terminated_items_count_ / 10),
                        static_cast<std::int64_t>(max_delete_count));

                thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    --terminated_items_count_;

                    thread_map_type::iterator it = thread_map_.find(todelete);

                    // this thread has to be in this map
                    HPX_ASSERT(it != thread_map_.end());

                    recycle_thread(*it);

                    thread_map_.erase(it);
                    --thread_map_count_;
                    HPX_ASSERT(thread_map_count_ >= 0);

                    --delete_count;
                }
            }
            return terminated_items_count_ == 0;
        }

        bool cleanup_terminated_locked(bool delete_all = false)
        {
            return cleanup_terminated_locked_helper(delete_all) &&
                thread_map_.empty();
        }

    public:
        bool cleanup_terminated(bool delete_all = false)
        {
            if (terminated_items_count_ == 0)
                return thread_map_count_ == 0;

            if (delete_all) {
                // do not lock mutex while deleting all threads, do it piece-wise
                bool thread_map_is_empty = false;
                while (true)
                {
                    std::lock_guard<mutex_type> lk(mtx_);
                    if (cleanup_terminated_locked_helper(false))
                    {
                        thread_map_is_empty =
                            (thread_map_count_ == 0) && (new_tasks_count_ == 0);
                        break;
                    }
                }
                return thread_map_is_empty;
            }

            std::lock_guard<mutex_type> lk(mtx_);
            return cleanup_terminated_locked_helper(false) &&
                (thread_map_count_ == 0) && (new_tasks_count_ == 0);
        }

        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

        edf_queue(std::size_t queue_num = std::size_t(-1),
                std::size_t max_count = max_thread_count)
          : min_tasks_to_steal_pending(detail::get_min_tasks_to_steal_pending()),
            min_tasks_to_steal_staged(detail::get_min_tasks_to_steal_staged()),
            min_add_new_count(detail::get_min_add_new_count()),
            max_add_new_count(detail::get_max_add_new_count()),
            max_delete_count(detail::get_max_delete_count()),
            max_terminated_threads(detail::get_max_terminated_threads()),
            thread_map_count_(0),
            work_items_(&work_items_ordering),
            work_items_count_(0),
            terminated_items_(128),
            terminated_items_count_(0),
            max_count_((0 == max_count)
                      ? static_cast<std::size_t>(max_thread_count)
                      : max_count),
            new_tasks_(128),
            new_tasks_count_(0),
            memory_pool_(64),
            thread_heap_small_(),
            thread_heap_medium_(),
            thread_heap_large_(),
            thread_heap_huge_(),
            add_new_logger_("edf_queue::add_new")
        {}

        void set_max_count(std::size_t max_count = max_thread_count)
        {
            max_count_ = (0 == max_count) ? max_thread_count : max_count; //-V105
        }

        ///////////////////////////////////////////////////////////////////////
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length() const
        {
            return work_items_count_ + new_tasks_count_;
        }

        // This returns the current length of the pending queue
        std::int64_t get_pending_queue_length() const
        {
            return work_items_count_;
        }

        // This returns the current length of the staged queue
        std::int64_t get_staged_queue_length(
            std::memory_order order = std::memory_order_seq_cst) const
        {
            return new_tasks_count_.load(order);
        }

        void increment_num_pending_misses(std::size_t num = 1) {}
        void increment_num_pending_accesses(std::size_t num = 1) {}
        void increment_num_stolen_from_pending(std::size_t num = 1) {}
        void increment_num_stolen_from_staged(std::size_t num = 1) {}
        void increment_num_stolen_to_pending(std::size_t num = 1) {}
        void increment_num_stolen_to_staged(std::size_t num = 1) {}

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec)
        {
            // thread has not been created yet
            if (id) *id = invalid_thread_id;
                threads::thread_id_type thrd;

                // The mutex can not be locked while a new thread is getting
                // created, as it might have that the current HPX thread gets
                // suspended.
            {
                std::unique_lock<mutex_type> lk(mtx_);

                create_thread_object(thrd, data, initial_state, lk);

                // add a new entry in the map for this thread
                std::pair<thread_map_type::iterator, bool> p =
                    thread_map_.insert(thrd);

                if (HPX_UNLIKELY(!p.second)) {
                    HPX_THROWS_IF(ec, hpx::out_of_memory,
                        "threadmanager::register_thread",
                        "Couldn't add new thread to the map of threads");
                    return;
                }
                ++thread_map_count_;

                // this thread has to be in the map now
                HPX_ASSERT(thread_map_.find(thrd.get()) != thread_map_.end());
                HPX_ASSERT(thrd->get_pool() == &memory_pool_);

                // push the new thread in the pending queue thread
                if (initial_state == pending)
                    schedule_thread(thrd.get());

                // return the thread_id of the newly created thread
                if (id) *id = std::move(thrd);

                if (&ec != &throws)
                    ec = make_success_code();
                return;
            }
        }

        void move_work_items_from(edf_queue *src, std::int64_t count)
        {
            thread_description* trd;
            while (!src->work_items_.empty())
            {
                trd = src->work_items_.top();
                src->work_items_.pop();
                --src->work_items_count_;

                bool finished = count == ++work_items_count_;
                work_items_.push(trd);
                if (finished)
                    break;
            }
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(threads::thread_data*& thrd,
            bool allow_stealing = false, bool steal = false) HPX_HOT
        {
            std::int64_t work_items_count =
                work_items_count_.load(std::memory_order_relaxed);

            if (allow_stealing && min_tasks_to_steal_pending > work_items_count)
            {
                return false;
            }

            std::lock_guard<hpx::util::spinlock> lg(work_items_mutex_);
            if (0 != work_items_count && !work_items_.empty())
            {
                thrd = work_items_.top();
                work_items_.pop();
                --work_items_count_;
                next_deadline_ = work_items_.top()->get_deadline();
                return true;
            }
            return false;
        }
        
        void peek_next_thread(threads::thread_data*& thrd)
        {
            std::int64_t work_items_count =
            work_items_count_.load(std::memory_order_relaxed);
            std::lock_guard<hpx::util::spinlock> lg(work_items_mutex_);
            if (0 != work_items_count && !work_items_.empty())
            {
                thrd = work_items_.top();
            }
        }
        
        std::chrono::steady_clock::time_point peek_next_deadline()
        {
            return next_deadline_;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, bool other_end = false)
        {
            ++work_items_count_;
            std::lock_guard<hpx::util::spinlock> lg(work_items_mutex_);
            work_items_.push(thrd);
            next_deadline_ = work_items_.top()->get_deadline();
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, std::int64_t& busy_count)
        {
            if (thrd->get_pool() == &memory_pool_)
            {
                terminated_items_.push(thrd);

                std::int64_t count = ++terminated_items_count_;
                if (count > max_terminated_threads)
                {
                    cleanup_terminated(true);   // clean up all terminated threads
                }
                return true;
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the number of existing threads with the given state.
        std::int64_t get_thread_count(thread_state_enum state = unknown) const
        {
            if (terminated == state)
                return terminated_items_count_;

            if (staged == state)
                return new_tasks_count_;

            if (unknown == state)
                return thread_map_count_ + new_tasks_count_ - terminated_items_count_;

            // acquire lock only if absolutely necessary
            std::lock_guard<mutex_type> lk(mtx_);

            std::int64_t num_threads = 0;
            thread_map_type::const_iterator end = thread_map_.end();
            for (thread_map_type::const_iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it)->get_state().state() == state)
                    ++num_threads;
            }
            return num_threads;
        }

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            std::lock_guard<mutex_type> lk(mtx_);
            thread_map_type::iterator end =  thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin();
                 it != end; ++it)
            {
                if ((*it)->get_state().state() == suspended)
                {
                    (*it)->set_state(pending, wait_abort);
                    schedule_thread((*it).get());
                }
            }
        }

        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const
        {
            std::uint64_t count = thread_map_count_;
            if (state == terminated)
            {
                count = terminated_items_count_;
            }
            else if (state == staged)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "edf_queue::iterate_threads",
                    "can't iterate over thread ids of staged threads");
                return false;
            }

            std::vector<thread_id_type> ids;
            ids.reserve(static_cast<std::size_t>(count));

            if (state == unknown)
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    ids.push_back(*it);
                }
            }
            else
            {
                std::lock_guard<mutex_type> lk(mtx_);
                thread_map_type::const_iterator end =  thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin();
                     it != end; ++it)
                {
                    if ((*it)->get_state().state() == state)
                        ids.push_back(*it);
                }
            }

            // now invoke callback function for all matching threads
            for (thread_id_type const& id : ids)
            {
                if (!f(id))
                    return false;       // stop iteration
            }

            return true;
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        inline bool wait_or_add_new(bool running,
            std::int64_t& idle_loop_count, std::size_t& added,
            edf_queue* addfrom = nullptr, bool steal = false) HPX_HOT
        {
            // try to generate new threads from task lists, but only if our
            // own list of threads is empty
            if (0 == work_items_count_.load(std::memory_order_relaxed))
            {
                // see if we can avoid grabbing the lock below
                if (addfrom)
                {
                    // don't try to steal if there are only a few tasks left on
                    // this queue
                    if (running && min_tasks_to_steal_staged >
                        addfrom->new_tasks_count_.load(std::memory_order_relaxed))
                    {
                        return false;
                    }
                }
                else
                {
                    if (running &&
                        0 == new_tasks_count_.load(std::memory_order_relaxed))
                    {
                        return false;
                    }
                    addfrom = this;
                }

                // No obvious work has to be done, so a lock won't hurt too much.
                //
                // We prefer to exit this function (some kind of very short
                // busy waiting) to blocking on this lock. Locking fails either
                // when a thread is currently doing thread maintenance, which
                // means there might be new work, or the thread owning the lock
                // just falls through to the cleanup work below (no work is available)
                // in which case the current thread (which failed to acquire
                // the lock) will just retry to enter this loop.
                std::unique_lock<mutex_type> lk(mtx_, std::try_to_lock);
                if (!lk.owns_lock())
                    return false;            // avoid long wait on lock

                // stop running after all HPX threads have been terminated
                bool added_new = add_new_always(added, addfrom, lk, steal);
                if (!added_new) {
                    // Before exiting each of the OS threads deletes the
                    // remaining terminated HPX threads
                    // REVIEW: Should we be doing this if we are stealing?
                    bool canexit = cleanup_terminated_locked(true);
                    if (!running && canexit) {
                        // we don't have any registered work items anymore
                        //do_some_work();       // notify possibly waiting threads
                        return true;            // terminate scheduling loop
                    }
                    return false;
                }

                cleanup_terminated_locked();
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        bool dump_suspended_threads(std::size_t num_thread
          , std::int64_t& idle_loop_count, bool running)
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread) {}
        void on_stop_thread(std::size_t num_thread) {}
        void on_error(std::size_t num_thread, std::exception_ptr const& e) {}

    private:
        mutable mutex_type mtx_;                    ///< mutex protecting the members

        thread_map_type thread_map_;
        ///< mapping of thread id's to HPX-threads
        std::atomic<std::int64_t> thread_map_count_;
        ///< overall count of work items
        
        std::chrono::steady_clock::time_point next_deadline_;

        work_items_type work_items_;
        ///< list of active work items
        std::atomic<std::int64_t> work_items_count_;
        ///< count of active work items
        hpx::util::spinlock work_items_mutex_;
        
        terminated_items_type terminated_items_;     ///< list of terminated threads
        std::atomic<std::int64_t> terminated_items_count_;
        ///< count of terminated items

        std::size_t max_count_;
        ///< maximum number of existing HPX-threads
        task_items_type new_tasks_;
        ///< list of new tasks to run

        std::atomic<std::int64_t> new_tasks_count_;
        ///< count of new tasks to run

        threads::thread_pool memory_pool_;          ///< OS thread local memory pools for
                                                    ///< HPX-threads

        std::list<thread_id_type> thread_heap_small_;
        std::list<thread_id_type> thread_heap_medium_;
        std::list<thread_id_type> thread_heap_large_;
        std::list<thread_id_type> thread_heap_huge_;

        util::block_profiler<add_new_tag> add_new_logger_;
    };
}}}

#endif

