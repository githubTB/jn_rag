from core.dedup import DedupStore

store = DedupStore()

# 测试文件 hash
h = DedupStore.hash_file("uploads/2020年水电气汇总统计表.jpg")
print("hash:", h[:16])

# 第一次：不存在
print("exists:", store.file_exists(h))  # False

# 登记
store.add_file(h, "2020年水电气汇总统计表.jpg", "uploads/2020年水电气汇总统计表.jpg")
print("exists:", store.file_exists(h))  # True

# 统计
print(store.stats())
