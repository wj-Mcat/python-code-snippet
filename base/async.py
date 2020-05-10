#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :asyncio.py
@说明    :
@时间    :2020/03/07 14:48:50
@作者    :吴京京
@版本    :0.0.1
'''

import asyncio
import requests
from datetime import datetime

async def get_content(url: str, index: int) -> str:
    print(f"getting {index}")
    res = requests.get(url)
    await asyncio.sleep(1)
    print(f"getted {index}")
    if res.status_code == 200:
        return "ok"
    return "bad"

async def test_connection():
    url = "http://www.baidu.com"
    now = datetime.now()
    result = []
    for i in range(20):
        result.append(await get_content(url, i))
    print(result)
    print(datetime.now() - now)

async def run():
    """
    this method runs in sequence
    """
    await test_connection()

# asyncio.run(run())

async def get_connections():
    url = "http://www.baidu.com"
    result = []
    for i in range(20):
        result.append(await get_content(url, i))
    return result

async def run_gather():
    """
    runs in parallel
    """
    url = "http://www.baidu.com"
    result = []

    for i in range(20):
        result.append(get_content(url, i))

    await asyncio.gather(*result)

# asyncio.run(run_gather())

async def run_hello():
    print("hello")
    await asyncio.sleep(1)
    print("world")

asyncio(run_hello())
    
