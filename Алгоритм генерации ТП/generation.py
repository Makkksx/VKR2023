import random
import time
from collections import Counter
from functools import reduce
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform


class TPGeneration:
    def __init__(self, n_cols=20, n_rows=10, min_count_marks=2, max_count_marks=10, distr=1):
        if distr == 2:  # нормальное
            pdf_norm = norm.pdf(np.linspace(-1, 1, max_count_marks - min_count_marks + 1), loc=0, scale=1)
            vr = pdf_norm / sum(pdf_norm)
        else:  # равномерное
            vr = uniform.pdf(np.linspace(0, 1, max_count_marks - min_count_marks + 1), loc=0, scale=max_count_marks - min_count_marks + 1)
        if (sum(vr) - 1.0) > 0.0000001:
            raise Exception("sum(vr) must be <=1")
        vrn = [0] * min_count_marks + [int(i * n_cols) for i in vr]
        vrn[-1] = n_cols - sum(vrn) + vrn[-1]
        print(vrn)
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
                new_df.loc[len(new_df)] = [i + 1, j + 1]
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
        if len(counter) != self.n_cols:
            raise Exception(
                f"К-во отметок недостаточно для создания циклической таблицы, к-во покрытых столбцов:  {len(counter)}")

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
        return rows[0:marks_count]

    def generate(self):
        for marks_count in range(len(self.vrn)):
            for _ in range(self.vrn[marks_count]):
                if not self.candidates:
                    break
                random.shuffle(self.candidates)
                rows_cur = self._choose_rows(marks_count)
                self._fill_rows(rows_cur)
                if self.col_choice:
                    self._decomposition(rows_cur)
        self._post_process()
        return self.result


min_count_marks = 5
max_count_marks = 30
gen = TPGeneration(n_cols=1000, n_rows=50, min_count_marks=min_count_marks, max_count_marks=max_count_marks, distr=1)
time_start = time.time()
print(gen.generate())
print("time", time.time() - time_start)
gen.count_df()
# gen.print_df_table()
#gen.pretty_df().to_csv('gen_test.csv', header=None, index=False)
