use [CoverTable]
GO

IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'Cover_Table')
BEGIN
	PRINT 'NOT EXISTS'
    Create Table dbo.Cover_Table (Row Int, Col INT);
END


TRUNCATE TABLE dbo.Cover_Table;			 --очистим перед импортом

-- необходимо задать путь к файлу источнику и 
-- название файла для вывода ошибок (не должен существовать)

BULK INSERT dbo.Cover_Table
    FROM 'C:\Users\mvolo\PycharmProjects\pythonProject\csv\Cover_Table.csv'
    WITH
    (
    FIRSTROW = 1,
    FIELDTERMINATOR = ',',  --CSV field delimiter
    ROWTERMINATOR = '\n',   --Use to shift the control to next row
	ERRORFILE = 'C:\Users\mvolo\PycharmProjects\pythonProject\csv\errorstable_Errors.csv',
    TABLOCK
)
GO
