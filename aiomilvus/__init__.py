# -*- coding  :   utf-8 -*-
# @Author     :   zhaojiangbing
# @File       :   __init__.py.py
# @Software   :   PyCharm


from .collection import AioCollection
from .grpc_handler import AioGrpcHandler

grpc_handler = AioGrpcHandler(host="localhost", port=19530, max_size=64, timeout=1)
AioCollection.grpc_handler = grpc_handler
