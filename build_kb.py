"""
一键建库：把 kb/ 下的文档切块、嵌入、写入 chromadb 持久化目录。

用法：
  python build_kb.py                  # 默认读 ./kb，写 ./chroma_db
  python build_kb.py ./kb --reset    # 清空旧库后重建
  python build_kb.py ./kb --persist-dir ./my_db
"""

import argparse
import sys

from rag import build_from_directory, DEFAULT_PERSIST_DIR


def main():
    parser = argparse.ArgumentParser(description="为 RawAgent 构建本地知识库（向量索引）。")
    parser.add_argument("doc_dir", nargs="?", default="./kb", help="文档目录（默认 ./kb）")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="向量库持久化目录")
    parser.add_argument("--reset", action="store_true", help="建库前清空旧索引")
    args = parser.parse_args()

    print(f"正在从 {args.doc_dir} 建库 -> {args.persist_dir} ...")
    try:
        _, summary = build_from_directory(
            args.doc_dir, args.persist_dir, reset=args.reset
        )
    except Exception as e:
        print(f"建库失败：{e}")
        sys.exit(1)

    print("建库完成：")
    print(f"  文档文件数 : {summary['files']}")
    print(f"  切块数量   : {summary['chunks']}")
    print(f"  向量库总量 : {summary['before']} -> {summary['after']}")


if __name__ == "__main__":
    main()
