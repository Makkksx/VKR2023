import time

import pandas as pd


class GreedyAlgorithm:
    def __init__(self):
        self.time = None
        self.df = None
        self.result = set()

    def print_results(self):
        print("Итоговый результат")
        print(f"Покрытие: {self.result}")
        print(f"Размер покрытия: {len(self.result)}")
        print(f"Время: {self.time}")

    def solve(self, data):
        start_time = time.time()
        self.df = data.groupby("i").j.apply(set)
        while len(self.df) != 0:
            choose = self.df.apply(len).idxmax()
            choose_cols = self.df[choose]
            self.result.add(choose)
            self.df = self.df.drop(choose)
            self.df = self.df.apply(lambda x, cols=choose_cols: x - cols)
            self.df = self.df[self.df != set()]
        self.time = time.time() - start_time


# df = pd.read_csv("result.csv", header=None, names=['i', 'j'])
# df = pd.read_excel("Cover_Table.xlsx", header=None, names=['i', 'j'])
df = pd.read_csv("gen_test_160x1600.csv", header=None, names=['i', 'j'])

gen = GreedyAlgorithm()
gen.solve(df)
gen.print_results()
