from Supervisor import *

supervisor = Supervisor(10, 2, 2)
asyncio.run(supervisor.run_parallelizer())