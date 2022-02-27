# -*- coding  :   utf-8 -*-
# @Author     :   zhaojiangbing
# @File       :   collection.py
# @Software   :   PyCharm


"""实现miluvs集合模块, 部分方法支持异步."""

from pymilvus import Collection
from pymilvus.orm.search import SearchResult
from pymilvus.orm.mutation import MutationResult
from pymilvus.orm.prepare import Prepare
from pymilvus.orm.exceptions import (
    DataTypeNotMatchException,
    ExceptionsMessage,
    SchemaNotReadyException,
)


class AioCollection(Collection):
    grpc_handler = None

    def __init__(self, name, schema=None, shards_num=2, **kwargs):

        super().__init__(name=name, schema=schema, using="default",
                         shards_num=shards_num, **kwargs)

    def _get_connection(self):
        return self.grpc_handler

    async def asearch(self, data, anns_field, param, limit, expr=None, partition_names=None,
                      output_fields=None, timeout=None, round_decimal=-1, **kwargs):
        """search异步实现"""

        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(0, ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = await conn.asearch(self._name, data, anns_field, param, limit, expr,
                                 partition_names, output_fields, timeout, round_decimal, **kwargs)

        return SearchResult(res)

    async def adelete(self, expr, partition_name=None, timeout=None, **kwargs):
        """delete异步实现."""

        conn = self._get_connection()
        res = await conn.adelete(self._name, expr, partition_name, timeout, **kwargs)
        return MutationResult(res)

    async def aquery(self, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
        """query异步版本."""

        if not isinstance(expr, str):
            raise DataTypeNotMatchException(0, ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = await conn.aquery(self._name, expr, output_fields, partition_names, timeout, **kwargs)
        return res

    async def ainsert(self, data, partition_name=None, timeout=None, **kwargs):
        """insert异步版本实现."""

        if data is None:
            return MutationResult(data)
        if not self._check_insert_data_schema(data):
            raise SchemaNotReadyException(0, ExceptionsMessage.TypeOfDataAndSchemaInconsistent)
        conn = self._get_connection()
        entities = Prepare.prepare_insert_data(data, self._schema)
        res = await conn.abulk_insert(self._name, entities, partition_name, ids=None, timeout=timeout, **kwargs)

        return MutationResult(res)
