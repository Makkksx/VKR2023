import random
import time
from collections import Counter
from functools import reduce
from math import sqrt

import pandas as pd


class TPGeneration:
    def __init__(self, n_cols=20, n_rows=10, vr=None):
        if vr is None:
            vr = [0.0, 0.3, 0.4, 0.3]
        if sum(vr) > 1:
            raise Exception("sum(vr) must be <=1")
        vrn = [0] + [int(i * n_cols) for i in vr]
        vrn[-1] = n_cols - sum(vrn) + vrn[-1]
        self.vrn = vrn
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.result = pd.DataFrame({"cols": [set() for _ in range(n_rows)], "cols_count": [0] * n_rows})
        self.candidates = [set(range(n_rows))]
        self.col_choice = list(range(n_cols))
        random.shuffle(self.col_choice)

    def pretty_df(self):
        new_df = pd.DataFrame({"Row_": [], "Col_": []})
        for i in range(len(self.result.cols)):
            for j in self.result.loc[i, "cols"]:
                new_df.loc[len(new_df)] = [i, j]
        return new_df

    def count_df(self):
        res = []
        for i in self.result.cols:
            res = res + list(i)
        res.sort()
        counter = Counter(res)
        print(counter)
        print("len", len(counter))
        print("count", sum(len(i) for i in self.result.cols))

    def print_df_table(self):
        for cols in self.result.cols:
            print([1 if i in cols else 0 for i in range(self.n_rows)])

    def _update_candidates(self, new_candidates, new_candidate):
        if len(new_candidate) >= 2 and not any(j >= new_candidate for j in new_candidates):
            new_candidates.append(new_candidate)

    def _post_process(self):
        self.result['cols_count'] = self.result['cols'].apply(len)

    def _decomposition(self, rows_cur):
        new_candidates = []
        need_cols_in_candidates = set(
            i for i in range(len(self.result.cols_count)) if not pd.isnull(self.result.cols_count[i]))
        while self.candidates:
            candidate = next((i for i in self.candidates if not i.isdisjoint(need_cols_in_candidates)), None)
            if not candidate:
                if len(new_candidates) >= sqrt(self.n_cols):
                    break
                else:
                    candidate = max(self.candidates, key=len)
            self.candidates.remove(candidate)
            if set(rows_cur) <= candidate:
                for el in rows_cur:
                    new_candidate = candidate.copy()
                    new_candidate.discard(el)
                    self._update_candidates(new_candidates, new_candidate)
            else:
                self._update_candidates(new_candidates, candidate)
            need_cols_in_candidates = need_cols_in_candidates - (reduce(lambda x, y: x | y,
                                                                        new_candidates) if new_candidates else set())
            if (not need_cols_in_candidates) and (len(new_candidates) >= sqrt(self.n_cols)):
                break
        # print(len(new_candidates), len(self.col_choice))
        self.candidates = new_candidates

    def _fill_rows(self, rows_cur):
        i_cur = self.col_choice.pop(0)
        self.result.loc[rows_cur, 'cols_count'] += 1
        self.result.loc[rows_cur, 'cols'].apply(lambda x: x.add(i_cur))

    def _choose_rows(self, marks_count):
        while True:
            str_need = self.result.cols_count.idxmin()
            candidates = [list(i) for i in self.candidates if str_need in i]
            if candidates:
                break
            self.result.loc[str_need, 'cols_count'] = None
        rows = next((i for i in candidates if len(i) >= marks_count), None)
        if rows is None:
            return max(candidates, key=len)
        random.shuffle(rows)
        rows.remove(str_need)
        rows = [str_need] + rows
        # rows.sort(key=lambda x: self.result.cols_count[x])
        return rows[0:marks_count]

    def generate(self):
        for marks_count in range(len(self.vrn)):
            for _ in range(self.vrn[marks_count]):
                if not self.candidates:
                    break
                random.shuffle(self.candidates)
                # self.candidates.sort(key=len, reverse=True)
                rows_cur = self._choose_rows(marks_count)
                self._fill_rows(rows_cur)
                if self.col_choice:
                    self._decomposition(rows_cur)
        self._post_process()
        return self.result


random.seed(11)
# n_cols > n_rows
gen = TPGeneration(n_cols=5000, n_rows=120, vr=[0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
time_start = time.time()
print(gen.generate())
print("time", time.time() - time_start)
gen.count_df()
# gen.print_df_table()
# gen.pretty_df().to_csv('result.csv', header=None, index=False)
