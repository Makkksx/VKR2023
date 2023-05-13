use [CoverTable]

DECLARE
    @n INT,
    @m INT,
    @min_count_marks INT,
    @max_count_marks INT,
    @distr INT
SET @n = 100
SET @m = 50
SET @min_count_marks = 2
SET @max_count_marks = 10
SET @distr = 1

EXECUTE Create_Algorithm @n, @m, @min_count_marks, @max_count_marks, @distr
