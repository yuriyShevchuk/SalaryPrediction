Задача предсказать годовой оклад исходя из описания вакансии.
Для её решения будем использовать линейную регрессию с L2-регуляризацией т.к. линейные алгоритмы хорошо подходят для работы с разряженными 
данными. Для преобразования текстов в векторы признаков будем использовать TfidfVectorizer. Признаки  LocationNormalized и ContractTime 
являются категориальными, поэтому для работы с ними будем использовать one-hot-кодирование (DictVectorizer).
