USE [CoverTable]
GO

IF OBJECT_ID ( 'Ant_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE Ant_Algorithm;
	PRINT 'DELETE';
GO


CREATE PROCEDURE Ant_Algorithm  (
		@tableName VARCHAR(80),					-- название таблицы с покрытием (например "Cover_Table")
		@resultCost INT OUTPUT,					
		@evaporationCoef FLOAT = 0.2,							
		@iterNum INT = 2,
		@pheromoneCoef FLOAT = 1.0,
		@heuristicCoef FLOAT = 1.0,
		@pheromoneNum INT = 1,
		@eliteActive BIT = 0,
		@eliteFreq INT = 3,
		@antsNum INT
)
AS

DECLARE @requestTable NVARCHAR(max)

SET @requestTable = 'SELECT ['+@tableName+'].Row, ['+@tableName+'].Col FROM ['+@tableName+']';
SET @resultCost = 0;


BEGIN
	-- создать таблицу результат (просто набор номеров вошедших рядов)
	IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Ant_Algo_Result')
	BEGIN
		PRINT 'NOT EXISTS'
		Create Table Ant_Algo_Result (Row_ INT);
	END
	TRUNCATE TABLE Ant_Algo_Result;


	INSERT INTO Ant_Algo_Result 
	EXECUTE sp_execute_external_script @language = N'Python',
	@script = N'
import pandas as pd
import random
import time
import numpy as np


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
        self.vc = None
        self.vs = None
        self.va = None
        self.df = None
        self.n_rows = None
        self.t = None
        self.p = p
        self.N = n
        self.a = a
        self.b = b
        self.f = f
        self.elite_active = elite_active
        self.elite_freq = elite_freq
        self.k = k

    # Количество вершин что строка может покрыть
    # Количество единиц в строке i
    def cost(self, i):
        return len(self.df.loc[i])

    # Количество новых вершин которые перекроются строкой i
    def cover(self, i):
        return len(self.vc - self.df.loc[i])

    # Эвристика / Аналог обратного веса ребра
    # Нулевые строки удаляются, так что всегда корректно
    def m(self, i):
        return self.cover(i) / self.cost(i)

    # Привлекательность вершины
    def attr(self, i):
        return (self.t[i] ** self.a) * (self.m(i) ** self.b)

    # Выбор вершины
    def choice(self):
        weight_sum = sum(self.attr(i) for i in self.va)
        return random.choices(tuple(self.va), weights=[self.attr(i) / weight_sum for i in self.va])[0]

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

    def set_df(self, data):
        self.df = data.groupby("i").j.apply(frozenset)
        self.n_rows = self.df.shape[0]
        self.rows = list(self.df.keys())
        self.n_cols = len(set(data.j))
        self.t = pd.Series(data=self.f, index=self.rows)
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
        for i in range(self.N):
            best = self.rows
            ants = random.choices(self.rows, k=self.k)
            for ant in ants:
                self.va = self.start_va(ant)
                self.vs = self.start_vs(ant)
                self.vc = self.start_vc(ant)
                while len(self.vc) != self.n_cols:
                    next_v = self.choice()
                    if self.cover(next_v) != 0:
                        self.vs.add(next_v)
                        self.vc |= self.df.loc[next_v]
                    self.va.remove(next_v)
                if len(self.vs) < len(best):
                    best = self.vs
            c = len(best)
            c_best = len(self.result)
            if c < c_best:
                self.result = best
                c_best = c
                self.print_inter_result(i)

            dt = pd.Series(data=(1 / (1 - (c_best - c) / c_best) if i in best else 0 for i in self.t.keys()),
                           index=self.rows)
            self.t = pd.Series(data=(self.t[i] * (1 - self.p) + dt[i] for i in self.t.keys()), index=self.rows)
            # Элитный муравей
            if self.elite_active:
                if (i + 1) % self.elite_freq == 0:
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
print(k)
elite_active = eliteActive
elite_freq = eliteFreq
InputDataSet.columns= ["i", "j"]
gen = AntAlgorithm(p, n, a, b, f, elite_active, elite_freq, k)
gen.solve(InputDataSet)
gen.print_results()
optimum = pd.DataFrame(list(gen.result), columns=["Row_"])
resultCost = len(optimum)
OutputDataSet = optimum

	'
	, @input_data_1 = @requestTable
	, @params = N'@resultCost INT OUTPUT, @evaporationCoef FLOAT, @iterNum INT, @pheromoneCoef FLOAT, @heuristicCoef FLOAT, @pheromoneNum INT, @eliteActive BIT, @eliteFreq INT, @antsNum INt'
	, @resultCost = @resultCost OUTPUT
	, @evaporationCoef = @evaporationCoef	
	, @iterNum  =@iterNum
	, @pheromoneCoef = @pheromoneCoef
	, @heuristicCoef = @heuristicCoef
	, @pheromoneNum = @pheromoneNum	
	, @eliteActive = @eliteActive
	, @eliteFreq = @eliteFreq
	, @antsNum = @antsNum
END;
