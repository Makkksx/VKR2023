USE [CoverTable]
GO

IF OBJECT_ID ( 'Ant_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE Ant_Algorithm;
	PRINT 'DELETE';
GO

CREATE PROCEDURE Ant_Algorithm(
    @evaporationCoef FLOAT = 0.2,
    @iterNum INT = 2,
    @pheromoneCoef FLOAT = 1.0,
    @heuristicCoef FLOAT = 1.0,
    @pheromoneNum INT = 1,
    @antsNum INT,
    @n INT = 1, -- фиктивный параметр
    @m INT = 1 -- фиктивный параметр
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
import random
import time
import warnings

import numpy as np
import pandas as pd
from itertools import chain

warnings.simplefilter(action="ignore", category=FutureWarning)


class AntAlgorithm:
    # p - Коэффициент испарения
    # N - Количество итераций
    # a - Значимость феромона / "кучность" алгоритма
    # b - Значимость эвристики / "жадность" алгоритма
    # f - Начальное количество феромонов на вершинах
    # k - Количество муравьёв
    def __init__(self, p=0.2, n=2, a=1.0, b=1.0, f=1, elite_active=False, elite_freq=3, k=None):
        self.rows = None
        self.time = None
        self.result = None
        self.n_cols = None
        self.df = None
        self.n_rows = None
        self.t = None
        self.ta = None
        self.p = p
        self.N = n
        self.a = a
        self.b = b
        self.f = f
        self.iter_to_stop = int(n / 3)
        self.elite_active = elite_active
        self.elite_freq = elite_freq
        self.k = k
        self.move_arr = pd.DataFrame()

    # Количество вершин что строка может покрыть
    # Количество единиц в строке i
    def cost(self, i):
        return self.df.loc[i].apply(len)

    # Количество новых вершин которые перекроются строкой i
    def cover(self, i, vc):
        return (vc - self.df.loc[i]).apply(len)

    # Эвристика / Аналог обратного веса ребра
    # Нулевые строки удаляются, так что всегда корректно
    def m(self, i, vc):
        return self.cover(i, vc) / self.cost(i)

    # Привлекательность вершины
    def attr(self, i, vc):
        return self.ta[i] * (self.m(i, vc) ** self.b)

    # Шаг всех муравьев
    def ant_move(self):
        for index, row in self.move_arr.iterrows():
            attr = self.attr(row.va, row.vc)
            row.weights = attr / sum(attr)
            row.choices = random.choices(tuple(row.va), weights=row.weights, k=len(row.va))
            row.next_v = next((i for i in row.choices if len(row.vc - self.df.loc[i]) != 0), None)
            row.vs.add(row.next_v)
            row.vc |= self.df.loc[row.next_v]
            row.va.remove(row.next_v)
            if len(row.vc) == self.n_cols:
                break

    # Начальное решение
    def start_vs(self, i):
        return {i}

    # Начальное покрытие строкой i
    def start_vc(self, i):
        return set(self.df.loc[i])

    # Начальные индексы доступных вершин
    def start_va(self, i):
        ind = set(self.df.keys())
        ind.remove(i)
        return ind

    def reduce_vs(self, vs):
        for i in vs.copy():
            vsi = vs.copy()
            vsi.remove(i)
            liv = set(chain.from_iterable(self.df.loc[vsi].values))
            iv = self.df.loc[i]
            s = iv - liv
            if not s:
                vs.remove(i)

    def set_df(self, data):
        self.df = data.groupby("i").j.apply(frozenset)
        self.n_rows = self.df.shape[0]
        self.rows = list(self.df.keys())
        self.n_cols = len(set(data.j))
        self.t = pd.Series(data=self.f, index=self.rows)
        self.ta = self.t ** self.a
        if self.k is None:
            self.k = int(np.sqrt(self.n_rows))

    def print_results(self):
        print(f"Итоговый результат")
        print(f"Покрытие: {self.result}")
        print(f"Размер покрытия: {len(self.result)}")
        print(f"Количество итераций: {self.N}")
        print(f"Количество муравьёв: {self.k}")
        print(f"Время: {self.time}")

    def print_inter_result(self, iter):
        print(f"Итерация: {iter}")
        print(f"Покрытие: {self.result}")
        print(f"Размер покрытия: {len(self.result)}\n")

    def solve(self, data):
        start_time = time.time()
        self.set_df(data)
        self.result = self.rows
        for iteration in range(self.N):
            print(iteration)
            best = self.rows
            ants = random.choices(self.rows, k=self.k)
            self.move_arr["va"] = [self.start_va(ant) for ant in ants]
            self.move_arr["vs"] = [self.start_vs(ant) for ant in ants]
            self.move_arr["vc"] = [self.start_vc(ant) for ant in ants]
            while max(self.move_arr.vc.apply(len)) != self.n_cols:
                self.ant_move()
            ant = self.move_arr.vc.apply(len).idxmax()
            best = self.move_arr.loc[ant].vs
            self.reduce_vs(best)
            c = len(best)
            c_best = len(self.result)
            if c < c_best:
                self.result = best
                c_best = c
                self.print_inter_result(iteration)
                alg_stop = 0
            else:
                alg_stop += 1
                if alg_stop >= self.iter_to_stop:
                    break
            dt = pd.Series(data=(1 / (1 - (c_best - c) / c_best) if i in best else 0 for i in self.t.keys()),
                           index=self.rows)
            self.t = pd.Series(data=(self.t[i] * (1 - self.p) + dt[i] for i in self.t.keys()), index=self.rows)
            self.ta = self.t ** self.a
            # Элитный муравей
            if self.elite_active:
                if (iteration + 1) % self.elite_freq == 0:
                    self.t = pd.Series(
                        data=(self.t[i] + 1 if i in self.result else self.t[i] for i in self.t.keys()),
                        index=self.rows)
        self.time = time.time() - start_time

p = evaporationCoef
n = iterNum
a = pheromoneCoef
b = heuristicCoef
f = pheromoneNum
k=None
if antsNum>0:
	k = antsNum
elite_active = False
elite_freq = 0
InputDataSet.columns= ["i", "j"]
gen = AntAlgorithm(p, n, a, b, f, elite_active, elite_freq, k)
gen.solve(InputDataSet)
gen.print_results()
optimum = pd.DataFrame(list(gen.result), columns=["Row_"])
OutputDataSet = optimum

	'
            , @input_data_1 = @requestTable
            ,
                @params = N'@m INT, @n INT, @antsNum INT, @iterNum INT, @evaporationCoef FLOAT, @pheromoneCoef FLOAT, @heuristicCoef FLOAT, @pheromoneNum INT'
            , @evaporationCoef = @evaporationCoef
            , @iterNum =@iterNum
            , @pheromoneCoef = @pheromoneCoef
            , @heuristicCoef = @heuristicCoef
            , @pheromoneNum = @pheromoneNum
            , @antsNum = @antsNum
            , @m = @m
            , @n = @n
    exec [cts].[fill_decision]
END;

