# -*- coding  :   utf-8 -*-
# @Author     :   zhaobao
# @File       :   zhaojiangbing.py
# @Software   :   PyCharm


import copy
import grpc
import asyncio

from grpc._cython import cygrpc
from pymilvus.client.prepare import Prepare
from pymilvus.client.check import check_pass_param
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client import ts_utils
from pymilvus.client.abstract import CollectionSchema, ChunkedQueryResult, MutationResult
from pymilvus.client.exceptions import DescribeCollectionException, ParamError
from pymilvus.client.types import (Status, DataType)
from pymilvus.client.utils import len_of, check_invalid_binary_vector
from .pool import Pool


class AioGrpcHandler(GrpcHandler):
    """实现grpc handler类."""

    _instance = None  # 用来实现单例模式

    def __init__(self, host="localhost", port=19530, max_size=64, timeout=1, **kwargs):

        self._pool = None
        self.timeout = timeout
        self.max_size = max_size
        uri = f"{host}:{port}"
        options = [(cygrpc.ChannelArgKey.max_send_message_length, -1),
                   (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                   ('grpc.enable_retries', 1),
                   ('grpc.keepalive_time_ms', 55000)]
        channel = grpc.insecure_channel(uri, options=options)  # 创建同步channel
        self.aiochannel = grpc.aio.insecure_channel(uri, options=options)  # 创建异步channel
        super().__init__(uri=uri, host=host, port=port, channel=channel, **kwargs)

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def pool(self):
        """获取连接池."""

        if self._pool is None:
            self._pool = Pool(channel=self.aiochannel, max_size=self.max_size)

        return self._pool

    async def abulk_insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        """bulk_insert异步版本实现."""

        if not check_invalid_binary_vector(entities):
            raise ParamError("Invalid binary vector data exists")

        timeout = self.timeout if timeout is None else timeout

        try:
            res = await self.adescribe_collection(collection_name, timeout, **kwargs)
            collection_id = res["collection_id"]
            request = self._prepare_bulk_insert_request(collection_name, entities, partition_name, timeout, **kwargs)

            async with self.pool.acquire() as stub:
                response = await stub.Insert(request, wait_for_ready=True, timeout=timeout)

            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_id, m.timestamp)
                return m

            raise BaseException(response.status.error_code, response.status.reason)
        except Exception as err:
            raise err

    async def adelete(self, collection_name, expression, partition_name=None, timeout=None, **kwargs):
        """delete异步版本实现."""

        check_pass_param(collection_name=collection_name)
        timeout = self.timeout if timeout is None else timeout

        try:
            res = await self.adescribe_collection(collection_name, timeout, **kwargs)
            collection_id = res["collection_id"]
            req = Prepare.delete_request(collection_name, partition_name, expression)

            async with self.pool.acquire() as stub:
                response = await stub.Delete(req, wait_for_ready=True, timeout=timeout)

            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_id, m.timestamp)
                return m

            raise BaseException(response.status.error_code, response.status.reason)
        except Exception as err:
            raise err

    async def aquery(self, collection_name, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
        """query异步版本实现."""

        if output_fields is not None and not isinstance(output_fields, (list,)):
            raise ParamError("Invalid query format. 'output_fields' must be a list")

        timeout = self.timeout if timeout is None else timeout
        collection_schema = await self.adescribe_collection(collection_name, timeout)
        collection_id = collection_schema["collection_id"]
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = kwargs.get("consistency_level", consistency_level)

        ts_utils.construct_guarantee_ts(consistency_level, collection_id, kwargs)
        guarantee_timestamp = kwargs.get("guarantee_timestamp", 0)
        travel_timestamp = kwargs.get("travel_timestamp", 0)

        request = Prepare.query_request(collection_name, expr, output_fields, partition_names, guarantee_timestamp,
                                        travel_timestamp)

        async with self.pool.acquire() as stub:
            response = await stub.Query(request, wait_for_ready=True, timeout=timeout)

        if response.status.error_code == Status.EMPTY_COLLECTION:
            return list()
        if response.status.error_code != Status.SUCCESS:
            raise BaseException(response.status.error_code, response.status.reason)

        num_fields = len(response.fields_data)
        # check has fields
        if num_fields == 0:
            raise BaseException(0, "")

        # check if all lists are of the same length
        it = iter(response.fields_data)
        num_entities = len_of(next(it))
        if not all(len_of(field_data) == num_entities for field_data in it):
            raise BaseException(0, "The length of fields data is inconsistent")

        # transpose
        results = list()
        for index in range(0, num_entities):
            result = dict()
            for field_data in response.fields_data:
                if field_data.type == DataType.BOOL:
                    raise BaseException(0, "Not support bool yet")
                    # result[field_data.name] = field_data.field.scalars.data.bool_data[index]
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    result[field_data.field_name] = field_data.scalars.int_data.data[index]
                elif field_data.type == DataType.INT64:
                    result[field_data.field_name] = field_data.scalars.long_data.data[index]
                elif field_data.type == DataType.FLOAT:
                    result[field_data.field_name] = round(field_data.scalars.float_data.data[index], 6)
                elif field_data.type == DataType.DOUBLE:
                    result[field_data.field_name] = field_data.scalars.double_data.data[index]
                elif field_data.type == DataType.STRING:
                    raise BaseException(0, "Not support string yet")
                    # result[field_data.field_name] = field_data.scalars.string_data.data[index]
                elif field_data.type == DataType.FLOAT_VECTOR:
                    dim = field_data.vectors.dim
                    start_pos = index * dim
                    end_pos = index * dim + dim
                    result[field_data.field_name] = [round(x, 6) for x in
                                                     field_data.vectors.float_vector.data[start_pos:end_pos]]
                elif field_data.type == DataType.BINARY_VECTOR:
                    dim = field_data.vectors.dim
                    start_pos = index * (int(dim / 8))
                    end_pos = (index + 1) * (int(dim / 8))
                    result[field_data.field_name] = field_data.vectors.binary_vector[start_pos:end_pos]
            results.append(result)

        return results

    async def asearch(self, collection_name, data, anns_field, param, limit,
                      expression=None, partition_names=None, output_fields=None,
                      timeout=None, round_decimal=-1, **kwargs):
        """search方法的异步版."""

        check_pass_param(
            limit=limit,
            round_decimal=round_decimal,
            anns_field=anns_field,
            search_data=data,
            partition_name_array=partition_names,
            output_fields=output_fields,
            travel_timestamp=kwargs.get("travel_timestamp", 0),
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0)
        )
        timeout = self.timeout if timeout is None else timeout

        _kwargs = copy.deepcopy(kwargs)
        collection_schema = await self.adescribe_collection(collection_name, timeout)
        collection_id = collection_schema["collection_id"]
        auto_id = collection_schema["auto_id"]
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = _kwargs.get("consistency_level", consistency_level)
        _kwargs["schema"] = collection_schema

        ts_utils.construct_guarantee_ts(consistency_level, collection_id, _kwargs)

        requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, expression,
                                                     partition_names, output_fields, round_decimal, **_kwargs)
        _kwargs.pop("schema")
        _kwargs["auto_id"] = auto_id
        _kwargs["round_decimal"] = round_decimal

        return await self._aexecute_search_requests(requests, timeout, **_kwargs)

    async def _aexecute_search_requests(self, requests, timeout=None, **kwargs):
        """"execute_search_requests方法的异步版."""

        raws = []
        auto_id = kwargs.get("auto_id", True)
        timeout = self.timeout if timeout is None else timeout

        async with self.pool.acquire() as stub:
            for request in requests:
                response = await stub.Search(request, wait_for_ready=True, timeout=timeout)

                if response.status.error_code != 0:
                    raise BaseException(response.status.error_code, response.status.reason)

                raws.append(response)

        round_decimal = kwargs.get("round_decimal", -1)
        return ChunkedQueryResult(raws, auto_id, round_decimal)

    async def adescribe_collection(self, collection_name, timeout=None, **kwargs):
        """describe_collection方法异步版."""

        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        timeout = self.timeout if timeout is None else timeout

        async with self.pool.acquire() as stub:
            response = await stub.DescribeCollection(request, wait_for_ready=True, timeout=timeout)
            status = response.status

        if status.error_code == 0:
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.error_code, status.reason)
