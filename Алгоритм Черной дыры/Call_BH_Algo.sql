use [CoverTable]


DECLARE
@popSize INT,
@numIter INT,
@vectorDist INT,
@transferFun INT

SET @vectorDist = 2     -- euclid / manhattan / cheb (варианты векторных расстояний)
SET @popSize = 40
SET @numIter = 250
SET @transferFun = 2

EXECUTE BH_Algorithm @popSize, @numIter, @vectorDist,@transferFun