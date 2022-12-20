use [CoverTable]


DECLARE 
@table1 VARCHAR(80), 
@resultCost INT,
@n INT,							
@m INT,
@vr nvarchar(MAX)


SET @table1 = 'Cover_Table'    -- название таблицы источника таблицы, формат: Row INT, Col INT
SET @n=1440
SET @m=120
SET @vr='0.0, 0.3, 0.4, 0.3'


EXECUTE Create_Algorithm @table1, @resultCost OUTPUT, @n, @m, @vr;


PRINT @resultCost;
SELECT * FROM Create_Algo_Result;
