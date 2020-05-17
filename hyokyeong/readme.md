**5/16**

1. Data: PULocation ID로 전처리해서 업로드함.
2. Main, CustomEmbedding, location, dataset 파일에 모두 location 부분 추가함.

**5/17**

1. Main, location Embedding 수정. list 에러수정

----------------------------------------------
Main 파일 돌렸을 때, h5py 에러발생

  0%|                                                                                                                                                                                                              | 0/20150 [00:00<?, ?it/sT 
raceback (most recent call last):
  File "c:/Users/Ann/Documents/nyc_with_transformer/kyohoon/main.py", line 191, in <module>
    main(args)
  File "c:/Users/Ann/Documents/nyc_with_transformer/kyohoon/main.py", line 115, in main
    for i, input_ in enumerate(tqdm(dataloader_dict[phase])):
  File "C:\Users\Ann\Anaconda3\lib\site-packages\tqdm\std.py", line 1107, in __iter__
    for obj in iterable:
  File "C:\Users\Ann\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py", line 278, in __iter__
    return _MultiProcessingDataLoaderIter(self)
  File "C:\Users\Ann\Anaconda3\lib\site-packages\torch\utils\data\dataloader.py", line 682, in __init__
    w.start()
  File "C:\Users\Ann\Anaconda3\lib\multiprocessing\process.py", line 112, in start
    self._popen = self._Popen(self)
  File "C:\Users\Ann\Anaconda3\lib\multiprocessing\context.py", line 223, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "C:\Users\Ann\Anaconda3\lib\multiprocessing\context.py", line 322, in _Popen
    return Popen(process_obj)
  File "C:\Users\Ann\Anaconda3\lib\multiprocessing\popen_spawn_win32.py", line 89, in __init__
    reduction.dump(process_obj, to_child)
  File "C:\Users\Ann\Anaconda3\lib\multiprocessing\reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
  File "C:\Users\Ann\Anaconda3\lib\site-packages\h5py\_hl\base.py", line 308, in __getnewargs__
    raise TypeError("h5py objects cannot be pickled")
TypeError: h5py objects cannot be pickled
