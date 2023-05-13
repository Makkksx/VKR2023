use [CoverTable]


DECLARE
    @popSize INT,
    @numIter INT,
    @vectorDist INT,
    @transferFun INT,
    @move_type INT,
    @gamma FLOAT,
    @gamma_alter INT,
    @betta_0 FLOAT,
    @alpha FLOAT

SET @vectorDist = 1 -- euclid / manhattan / cheb (варианты векторных расстояний)
SET @popSize = 40
SET @numIter = 100
SET @transferFun = 1
SET @move_type = 1 -- разновидность перемещения светлячков
SET @gamma = 1.0 -- параметры алгоритма (0;1]
SET @gamma_alter = 0 -- альтернативный gamma. Если 0, то старый, иначе gamma = gamma / maxDist^gamma_alter рекомендованы [1,2,3]
SET @betta_0 = 1.0 -- параметр алгоритма, значение [0:1]
SET @alpha = 1.0 -- параметр алгоритма, значение [0;1]

EXECUTE FFA_Algorithm @popSize, @numIter, @vectorDist, @transferFun, @move_type, @gamma, @gamma_alter, @betta_0, @alpha
