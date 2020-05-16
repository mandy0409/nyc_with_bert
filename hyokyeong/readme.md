1. Data: PULocation ID로 전처리해서 업로드함.
2. Main, CustomEmbedding, location, dataset 파일에 모두 location 부분 추가함.

----------------------------------------------
Main 파일 돌렸을 때, 에러발생

Traceback (most recent call last):
  File "main.py", line 172, in <module>
    main(args)
  File "main.py", line 32, in main
    'train': CustomDataset(data_list[0]),
IndexError: list index out of range
