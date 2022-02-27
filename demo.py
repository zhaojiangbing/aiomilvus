# -*- coding  :   utf-8 -*-
# @Author     :   zhaojiangbing
# @File       :   demo.py
# @Software   :   PyCharm


import time
import asyncio
import traceback

from aiomilvus import AioCollection, grpc_handler

collection_name = "book"
collection = AioCollection(name=collection_name)

data = [0.6505, 0.7919, 0.1218, 0.5667, 0.3578, 0.0529, 0.1464, 0.7852, 0.9767, 0.3759, 0.5428, 0.4838, 0.2153,
        0.3071, 0.7021, 0.8454, 0.24, 0.71, 0.8861, 0.2147]
book_ids = [1, 3, 5, 7]
atime = int(time.time())
dtime = int(time.time()) + 120


async def abc():
    try:

        res = await collection.ainsert([book_ids, [atime, atime, atime, atime], [dtime, dtime, dtime, dtime],
                                        [data, data, data, data]], partition_name="A")
        ctime = int(time.time())

        # 异步查询
        res = await collection.aquery(expr="atime < {} && dtime > {}".format(ctime, ctime),
                                      output_fields=["book_id", "atime", "dtime"],
                                      timeout=10, partition_names=["A"], consistency_level="CONSISTENCY_STRONG")
        res = await collection.asearch(data=[data], anns_field="book_intro", limit=500, timeout=1,
                                       expr="book_id in [1, 3, 5, 7]", partition_names=["A"])
    except Exception as e:
        traceback.print_exc()
        print(11111, e)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(abc())
