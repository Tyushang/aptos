#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from aptos_preprocessing import *

__author__ = u"Frank Jing"


def main():
    all_images = files_to_process(dataset_dir, CACHE_DIR)
    images_per_process = ceil(len(all_images) / cpu_count())

    pool = Pool()
    for i in range(cpu_count()):
        pool.apply_async(proc_entry_for_train, args=(all_images[i * images_per_process: (i + 1) * images_per_process],))
    pool.close()
    pool.join()

    ###
    import tarfile
    tar = tarfile.open(OUT_TAR_FILE, 'w:gz')
    tar.add(CACHE_DIR)
    print("================ content of tar file ================")
    tar.list()
    tar.close()


if __name__ == '__main__':
    from datetime import datetime
    tic = datetime.now()
    main()
    toc = datetime.now()
    print("Total time cost: %f s" % (toc - tic).total_seconds())