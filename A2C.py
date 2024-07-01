from Supervisor import *

supervisor = Supervisor(150, 2, 1)
asyncio.run(supervisor.run_parallelizer())