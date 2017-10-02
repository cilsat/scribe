#!/usr/bin/env python

import asyncio
import websockets


async def hello(ws, path):
    name = await ws.recv()
    print("< {}".format(name))

    greeting = "Hello {}!".format(name)
    await ws.send(greeting)
    print("> {}".format(greeting))


start_server = websockets.serve(hello, 'localhost', 6600)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
