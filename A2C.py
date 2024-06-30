from Supervisor import *

supervisor = Supervisor(300, 5, 5)
asyncio.run(supervisor.run_parallelizer())