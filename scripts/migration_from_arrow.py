import os
import lmdb
import pyarrow  # keep 0.14.0 & Python3.7 only for reading
import pickle


def rename(old_directory_path, new_directory_path):
    try:
        os.rename(old_directory_path, new_directory_path)
        print(f"Directory '{old_directory_path}' successfully renamed to '{new_directory_path}'")
    except OSError as e:
        print(f"Error renaming directory: {e}")


def migrate(src_dir, map_size=2 * 1024 ** 3):
    arrow_dir = src_dir + "_arrow"
    rename(src_dir, arrow_dir)
    pickle_dir = src_dir
    src_env = lmdb.open(arrow_dir, readonly=True, lock=False)
    dst_env = lmdb.open(pickle_dir, map_size=map_size)  # new DB
    with src_env.begin(write=False) as txn_src, dst_env.begin(write=True) as txn_dst:
        for key, value in txn_src.cursor():
            # read with old pyarrow
            obj = pyarrow.deserialize(value)
            # re-encode with pickle
            txn_dst.put(key, pickle.dumps(obj))
    print("Migration finished. New LMDB saved at data/ted_dataset/lmdb_val_pickle")

    env = lmdb.open(pickle_dir, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        key, value = next(cursor.iternext())
        obj = pickle.loads(value)
        print("First key:", key)
        print("Object type:", type(obj))
        print("Keys inside:", obj if isinstance(obj, dict) else "list")


if __name__ == "__main__":
    migrate("data/ted_dataset/lmdb_val", map_size=2 * 1024 ** 3)
    migrate("data/ted_dataset/lmdb_test", map_size=2 * 1024 ** 3)
    migrate("data/ted_dataset/lmdb_train", map_size=16 * 1024 ** 3)
