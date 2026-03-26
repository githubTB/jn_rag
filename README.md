

上传文件
    ↓
文件级去重（Dedup.is_file_new）← 在这里拦截，重复文件直接返回
    ↓
保存文件到磁盘
    ↓
解析（ExtractProcessor.extract）
    ↓
切片（DocChunker.chunk）
    ↓
Chunk 级去重（Dedup.filter_new_chunks）← 过滤重复内容
    ↓
向量化（Embedder.embed）
    ↓
入库（Milvus）
