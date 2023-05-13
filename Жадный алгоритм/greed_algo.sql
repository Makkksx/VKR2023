USE [CoverTable]
GO

IF OBJECT_ID ( 'Greedy_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE Greedy_Algorithm;
	PRINT 'DELETE';
GO

CREATE PROCEDURE Greedy_Algorithm(
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
import pandas as pd
import time


class GreedyAlgorithm:
    def __init__(self):
        self.time = None
        self.df = None
        self.result = None

    def print_results(self):
        print("Итоговый результат")
        print(f"Покрытие: {self.result}")
        print(f"Размер покрытия: {len(self.result)}")
        print(f"Время: {self.time}")

    def solve(self, data):
        start_time = time.time()
        self.df = data.groupby("i").j.apply(set)
        self.result = set()
        while len(self.df) != 0:
            choose = self.df.apply(len).idxmax()
            choose_cols = self.df[choose]
            self.result.add(choose)
            self.df = self.df.drop(choose)
            self.df = self.df.apply(lambda x, cols=choose_cols: x - cols)
            self.df = self.df[self.df != set()]
        self.time = time.time() - start_time


InputDataSet.columns= ["i", "j"]
gen = GreedyAlgorithm()
gen.solve(InputDataSet)
gen.print_results()
optimum = pd.DataFrame(list(gen.result), columns=["Row_"])
OutputDataSet = optimum

	'
            , @input_data_1 = @requestTable
            , @params = N''
    exec [cts].[fill_decision]
END;

