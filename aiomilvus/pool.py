# -*- coding  :   utf-8 -*-
# @Author     :   zhaojiangbing
# @File       :   pool.py
# @Software   :   PyCharm


"""实现连接池."""

import asyncio

from pymilvus.grpc_gen import milvus_pb2_grpc


class Pool(object):
    """实现基于stub的连接池."""

    def __init__(self, channel, max_size=64, loop=None):

        self.__closed = False
        self.__created = 0  # 记录创建了多少个stub
        self.__item_set = set()  # 存放创建的所有stub
        self.__items = asyncio.Queue()
        self.__lock = asyncio.Lock()  # 协程安全锁
        self.__max_size = max_size  # 最大的stub数
        self.loop = loop or asyncio.get_event_loop()
        self.channel = channel

    @property
    def size(self):

        return self.__items.qsize()

    @property
    def is_closed(self) -> bool:
        return self.__closed

    def acquire(self):
        if self.__closed:
            raise Exception("acquire operation on closed pool")

        return PoolItemContextManager(self)

    @property
    def _has_released(self):
        return self.__items.qsize() > 0

    @property
    def _is_overflow(self) -> bool:
        if self.__max_size:
            return self.__created >= self.__max_size or self._has_released
        return self._has_released

    async def _create_item(self):
        if self.__closed:
            raise Exception("create item operation on closed pool")

        async with self.__lock:
            if self._is_overflow:
                return await self.__items.get()

            item = milvus_pb2_grpc.MilvusServiceStub(self.channel)
            self.__created += 1
            self.__item_set.add(item)
            return item

    async def _get(self):
        if self.__closed:
            raise Exception("get operation on closed pool")

        if self._is_overflow:
            return await self.__items.get()

        return await self._create_item()

    def put(self, item):
        if self.__closed:
            raise Exception("put operation on closed pool")

        return self.__items.put_nowait(item)

    async def close(self):
        async with self.__lock:
            self.__closed = True

            for item in self.__item_set:
                del item

            self.channel.close()

    async def __aenter__(self) -> "Pool":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.__closed:
            return

        await asyncio.shield(self.close())


class PoolItemContextManager(object):
    __slots__ = "pool", "item"

    def __init__(self, pool):
        self.pool = pool
        self.item = None

    async def __aenter__(self):
        # noinspection PyProtectedMember
        self.item = await self.pool._get()
        return self.item

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.item is not None:
            self.pool.put(self.item)
