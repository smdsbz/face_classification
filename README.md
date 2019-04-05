# face_classification
Backup port for a random data process project - classifying faces in 8M pictures into folders, with absolutely no priori knowledge.

## Usage

> __Note:__  
> If you feel uncertain about something, feel free to dig into the
> sources, they are well-documented (I guess).  

依照次序执行以下脚本：  

__1. `extract_to_db.py`__  

顺序将 `.tsv` 中所有原始 base64 数据读出，并将计算所得的 encoding 存到 SQLite 数据库 `encodings.db` 中。数据表 __初始__ 定义如下：  

```python3
cur.execute('''
    CREATE TABLE IF NOT EXISTS encodings (
        id          integer     primary key     ,
        encoding    blob        not null        ,
        belong      integer
    )'''
)
```

后续因需要加了两列，见 __2. `append_orig_class.py`__。  

- 定义了一些命令行参数：
    - `--range`：用于指定起始与结束 index，会覆盖读取的断点。
    - `--no-op`：不做任何数据操作，用来数有多少张照片的。
    - `--chunk-size`：`pandas.read_csv:chunksize` 设大了可以减少向数据库 commit 的次数，设小了断点保存会更密集，自行取舍。

- 原始 `.tsv` 数据文件路径我写死了，懒得用命令行参数了，请自行修改源码。

> __Note:__  
> Encoding 过程是真的慢！请确保 `face_recognition` 依赖的 `dlib` 至少是带 AVX512 支持的，能有 GPU 支持更好！

__2. `append_orig_class.py`__  

当时给 `encodings` 表又加了两列：  
- `orig_class` - 用来存 `.tsv` 数据的第 0 列（好像是原数据集给的一个初始基础分类，__后续分类仅在这个基础分类里面进行__，手头没有原始数据了无法验证），在 __手动__ 给 `encoginds` 表 `ALTER` 后可以运行这个脚本初始化。
    > 其实后面好像也没有用到这一列数据。。。  
- `count` - 用来存该图片中人脸的个数，也需要手动 ALTER 出这一列。


__3. `classify_faces.py`__  

分类脚本。  

- 每一个基础分类（segment）都在 `segment_db/` 下有以 segment 名称/序号命名的数据库，该数据库的表 `seen_classes` 用于存放分类算法所需信息。
- 这里使用的分类算法的简单描述如下：  
    1. 从 `.tsv` 中顺序读取原始数据行
        - 读取单行 `pandas.read_csv` 的迭代器比从 SQLite 中读需要 I/O 更少
    2. 如果仍在同一 segment 中，则执行下一步；否则新建 segment 数据库，并执行下一步
    3. 从 `encodings.db` 中取出 encoding，与 `seen_classes` 中每项 __按命中频率次序__ 比较与该类典型 encoding 的 L2 距离，若小于阈值则认为命中（见 `face_recognition.compare_faces()`）
    4. 若命中，则增加 `seen_classes` 中对应行 `count`；否则新插入一行，令其 `first_id` 指向当前数据
    5. 按 `belong` 字段将原数据图像分类存到 `output/` 下对应文件夹中

> __Note:__  
> 1. 本来 `seen_classes` 表会定期将（平均）出现频率小于一定阈值的类清掉，否则噪声数据也会加入排序比较环节，增加运行时间。但后来被告知原始数据中有预分类，只需要在预分类里面分类就行了，此时性能瓶颈已经转移到 I/O 上面，所以为了尽量保持数据完整性就把这个机制删去了。
> 2. 由于 `seen_classes` 缩减机制被删去，最终输出虽然包含更多数据，但可能有很多文件夹中就一张或几张图。

__其他__  

`dump_folders.py`  

这是非 segment 版本的图片分类导出脚本，在运行过非 segment 版本的 `classify_faces.py` 之后使用。  

__非 segment 版本 `classify_faces.py` 没有留档！__  

这里给出重新实现指南：  
1. 去除 segment 比较
2. 删去每条记录后的 `dump_to_folder()` 调用
3. `seen_classes` 数据表存放到 `encodings.db` 中
4. 添加 `seen_classes` 定期缩减机制
    - 维护一个全局 clock，每处理一行数据 tick 一次
    - timeout 后删去 `count / clock` 小于阈值的行

- - - - - - - - - - - - - - - - - - - - - - - -

## Extra Notes

- 当时考虑过把 encoding 步骤并行化，后来发现 `face_recognition` 库的初始化步骤太耗时了，还不如单线程跑得快。  
    当然可以弄一个类似 encoding server 的东西，只初始化一次，之后都用 pipe 传 base64 就完事了，但那个时候时间比较紧，就懒得做了 :rofl:
- `dlib` 一定要有加速，至少 AVX 加速，最好 GPU。如果用 GPU 加速，得一次传一个 batch 进去让它算，不然内存访问会成瓶颈。
- 若考虑分类算法的鲁棒性，可以选择不在 `seen_classes` 表中存 `first_id`，naiively 假设该类的第一个 encoding 就是典型值；而存放一个实时更新的 encoding，若后期有新数据命中该类，则取 `(1 - tau) * old_encoding + tau * new_encoding` 作为该类新典型值。但这样做可能会增加 I/O 次数，务必注意！
- `classify_faces.py` 中有认为两个 encoding 是同一个人的阈值可以调。
- 对于一张图里面有多个人脸的情况，segment 版本 naiively 取数据库中存的 encoding list 中的第一个（后续可以考虑在往 `encodings` 表中存时，只存框最大的那个，`face_recognition` 有取框的 API）；`dump_folders.py` 脚本则直接丢弃这些数据（因为本来图片分辨率就不高，如果有几张脸的话大概率全是糊的）。
