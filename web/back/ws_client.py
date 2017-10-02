#!/usr/bin/env python

import asyncio
import websockets


async def hello():
    async with websockets.connect('ws://localhost:6600') as ws:
        name = input("What's your name mofo? ")
        await ws.send(name)
        print("> {}".format(name))

        greeting = await ws.recv()
        print("< {}".format(greeting))


asyncio.get_event_loop().run_until_complete(hello())
