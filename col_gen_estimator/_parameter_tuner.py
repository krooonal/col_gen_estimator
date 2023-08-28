
import math
import random


class RunningStat:
    def __init__(self):
        self.num_data_ = 0
        self.old_mean_ = 0.0
        self.new_mean_ = 0.0
        self.old_std_ = 0.0
        self.new_std_ = 0.0

    def clear(self):
        self.num_data_ = 0

    def push(self, x):
        self.num_data_ += 1

        if self.num_data_ == 1:
            self.old_mean_ = x
            self.new_mean_ = x
            self.old_std_ = 0.0
        else:
            self.new_mean_ = self.old_mean_ + \
                (x - self.old_mean_) / self.num_data_
            self.new_std_ = self.old_std_ + \
                (x - self.old_mean_) * (x - self.new_mean_)

            self.old_mean_ = self.new_mean_
            self.old_std_ = self.new_std_

    def num_data_values(self):
        return self.num_data_

    def mean(self):
        return self.new_mean_ if self.num_data_ > 0 else 0.0

    def variance(self):
        if self.num_data_ > 1:
            return self.new_std_ / (self.num_data_ - 1)
        return 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


class Parameter:
    def __init__(self, c_fac, name, seed):
        self.name_ = name
        self.values_ = []
        self.scores_ = []
        self.final_scores_ = []
        self.counts_ = []
        self.c_fac_ = c_fac
        self.current_index_ = 0
        self.total_counts_ = 0
        self.mt_ = random.Random(seed)
        self.explore_count_ = 10
        self.switch_flag_ = 0

    def set_switch_flag(self, value):
        self.switch_flag_ = value

    def set_explore_count(self, value):
        self.explore_count_ = value

    def adjust_score(self, score, index=None):
        if index == None:
            index = self.current_index_
        self.scores_[index].push(score)
        self.counts_[index] += 1
        self.total_counts_ += 1
        self.final_scores_[index] = self.scores_[
            index].mean() + (self.c_fac_ / self.counts_[index])

    def add_value(self, value):
        self.values_.append(value)
        self.scores_.append(RunningStat())
        self.final_scores_.append(5.0)
        self.counts_.append(0)

    def get_best_value(self):
        best_index = 0

        if self.explore_count_ > 0:
            if len(self.values_) == 2:
                best_index = 0
                if self.switch_flag_ & self.explore_count_ > 0:
                    best_index = 1
            else:
                best_index = self.mt_.randrange(len(self.values_))
            self.explore_count_ -= 1
        else:  # Exploit
            bucket = [i for i in range(
                len(self.values_)) if self.counts_[i] < 4]

            if not bucket:
                for i in range(len(self.values_)):
                    if self.final_scores_[i] > self.final_scores_[best_index]:
                        best_index = i

                for i in range(len(self.values_)):
                    #  Only converge if gains are significant.
                    if (
                        self.final_scores_[
                            i] + self.scores_[i].standard_deviation() / 10.0
                        >= self.final_scores_[best_index]
                    ):
                        bucket.append(i)

            # bucket is never empty. The best param is always in it.
            assert len(bucket) > 0
            if len(bucket) > 1:
                best_index = bucket[self.mt_.randrange(len(bucket))]
            else:
                best_index = bucket[0]

        self.current_index_ = best_index
        return self.values_[best_index]

    def print_stats(self):
        print(self.name_)
        for i in range(len(self.values_)):
            print(
                f"Value {self.values_[i]} count {self.counts_[i]} "
                f"Qscore {self.scores_[i].mean()}"
                f"Final score {self.final_scores_[i]}"
            )
