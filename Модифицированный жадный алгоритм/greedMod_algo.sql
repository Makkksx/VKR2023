USE [CoverTable]
GO

IF OBJECT_ID ( 'Greedy_Mod_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE Greedy_Mod_Algorithm;
	PRINT 'DELETE';
GO

CREATE PROCEDURE Greedy_Mod_Algorithm(
    @n INT = 1, --фиктивный параметр
    @m INT = 1  --фиктивный параметр
)
AS

DECLARE @requestTable NVARCHAR(max)

    SET @requestTable = 'SELECT [Cvt].J, [Cvt].I FROM [Cvt]';


BEGIN
    -- создать таблицу результат (просто набор номеров вошедших рядов)
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Algo_Result')
        BEGIN
            PRINT 'NOT EXISTS'
            Create Table Algo_Result
            (
                Row_ INT
            );
        END
    TRUNCATE TABLE Algo_Result;


    INSERT INTO Algo_Result
        EXECUTE sp_execute_external_script @language = N'Python',
                @script = N'
import time

import pandas as pd


class DecisionCvt:
    # n - число столбцов
    # m - число строк
    def __init__(self):
        self.n_cols = None  # n
        self.n_rows = None  # m
        self.time = None
        self.df = None
        self.strings = None
        self.columns = None
        self.decision = set()
        self.eps = 0.0000001

    def column_del(self):
        self.columns.sort_values(by=["Kotm"], inplace=True, ascending=False)
        unchecked = list(self.columns.index[::-1])
        for j1 in self.columns.index:
            unchecked.pop()
            for j2 in unchecked:
                mj1 = set(self.df.i[self.df.j == j1])
                mj2 = set(self.df.i[self.df.j == j2])
                if mj1 >= mj2:
                    self.df.drop(self.df.index[self.df.j == j1], inplace=True)
                    self.columns.drop([j1], inplace=True)
                    break

    def string_del(self):
        self.strings.sort_values(by=["Kotm"], inplace=True, ascending=True)
        unchecked = list(self.strings.index[::-1])
        for i1 in self.strings.index:
            unchecked.pop()
            for i2 in unchecked:
                mi1 = set(self.df.j[self.df.i == i1])
                mi2 = set(self.df.j[self.df.i == i2])
                if mi1 <= mi2:
                    self.df.drop(self.df.index[self.df.i == i1], inplace=True)
                    self.strings.drop([i1], inplace=True)
                    break

    def string_choice(self):
        self.strings.sort_values(by=["f1"], inplace=True, ascending=False)
        for i1 in self.strings.index:
            mi1 = set(self.df.j[self.df.i == i1])
            i1f1 = self.strings.loc[i1].f1
            find_vec = lambda x: x in mi1
            self.decision.add(i1)
            self.strings.drop([i1], inplace=True)
            self.columns.drop(mi1, inplace=True)
            self.df.drop(self.df.index[self.df.j.apply(find_vec)], inplace=True)
            if i1f1 < 1 / self.eps:
                break

    def set_df(self, data):
        self.df = data.copy()
        self.n_rows = data.i.max()
        self.n_cols = data.j.max()
        self.strings = pd.DataFrame(index=data.i.unique())
        self.columns = pd.DataFrame(index=sorted(data.j.unique()))

    def set_Kotm_f1(self):
        self.strings["Kotm"] = [len(self.df[self.df.i == i]) for i in self.strings.index]
        self.columns["Kotm"] = [len(self.df[self.df.j == j]) for j in self.columns.index]
        self.columns["f1"] = [1.0 / (self.columns.loc[j].Kotm - 1 + self.eps) for j in self.columns.index]
        self.strings["f1"] = [sum(self.columns.loc[j].f1 for j in self.df.j[self.df.i == i]) for i in
                              self.strings.index]

    def print_results(self):
        # print(f"m: {self.n_rows}")
        # print(f"n: {self.n_cols}")
        print(f"Время: {self.time}")
        # print(f"ТП: {self.df}")
        # print(f"Строки: {self.strings}")
        # print(f"Столбцы: {self.columns}")
        print(f"Размер покрытия: {len(self.decision)}")
        print(f"Решение: {self.decision}")

    def solve(self, data):
        start_time = time.time()
        self.set_df(data)
        self.set_Kotm_f1()
        while self.columns.size:
            self.set_Kotm_f1()
            #self.column_del()
            #self.set_Kotm_f1()
            #self.string_del()
            self.string_choice()
        self.time = time.time() - start_time


InputDataSet.columns= ["i", "j"]
gen = DecisionCvt()
gen.solve(InputDataSet)
gen.print_results()
optimum = pd.DataFrame(list(gen.decision), columns=["Row_"])
OutputDataSet = optimum

	'
            , @input_data_1 = @requestTable
            , @params = N''

    exec [cts].[fill_decision]
END;

