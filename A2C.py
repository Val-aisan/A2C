from Supervisor import *

supervisor = Supervisor(10, 1, 1)
asyncio.run(supervisor.run_parallelizer())