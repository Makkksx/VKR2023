use [CoverTable]


DECLARE
    @evaporationCoef FLOAT,
    @iterNum INT,
    @pheromoneCoef FLOAT,
    @heuristicCoef FLOAT,
    @pheromoneNum INT,
    @antsNum INT


SET @evaporationCoef = 0.2
SET @iterNum = 5
SET @pheromoneCoef = 1.0
SET @heuristicCoef = 1.0
SET @pheromoneNum = 1
SET @antsNum = 14


EXECUTE Ant_Algorithm @evaporationCoef, @iterNum, @pheromoneCoef,@heuristicCoef,@pheromoneNum,@antsNum,120 ,40


SELECT *
FROM Algo_Result;
