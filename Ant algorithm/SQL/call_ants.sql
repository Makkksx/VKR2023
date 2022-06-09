use [CoverTable]


DECLARE 
@table1 VARCHAR(80), 
@resultCost INT,
@evaporationCoef FLOAT,
@iterNum INT,
@pheromoneCoef FLOAT,
@heuristicCoef FLOAT,
@pheromoneNum INT,
@eliteActive BIT,
@eliteFreq INT,
@antsNum INT


SET @table1 = 'Cover_Table'    -- название таблицы источника таблицы, формат: Row INT, Col INT
SET @evaporationCoef=0.2
SET @iterNum=50
SET @pheromoneCoef=1.0
SET @heuristicCoef=1.0
SET @pheromoneNum=1
SET @eliteActive=0
SET @eliteFreq=3
SET @antsNum=NULL


EXECUTE Ant_Algorithm @table1 , @resultCost OUTPUT, @evaporationCoef, @iterNum, @pheromoneCoef, @heuristicCoef, @pheromoneNum, @eliteActive, @eliteFreq, @antsNum;


PRINT @resultCost;
SELECT * FROM Ant_Algo_Result;
