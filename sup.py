from Supervisor import *

supervisor = Supervisor(100, 2, 2)
asyncio.run(supervisor.run_parallelizer())