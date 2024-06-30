from Supervisor import *

supervisor = Supervisor(6, 2, 2)
asyncio.run(supervisor.run_parallelizer())