import asyncio
import websockets
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocket:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.done = False
        self.reward1 = 0
        self.reward2 = 0
        self.score1 = 0
        self.score2 = 0
        self.blocked1 = 0
        self.blocked2 = 0
        self.state = None

    async def create_connection(self):
        try:
            self.ws = await websockets.connect(self.ws_url)
            asyncio.create_task(self.receive_data())
            print("WebSocket connection established")
        except websockets.WebSocketException as e:
            print(f"Failed to establish WebSocket connection: {e}")
            self.ws = None

    async def start_new_game(self):
        await self.ws.send(json.dumps({"t":"select_game_type","game_type":"solo","username":"dd"}))
        await self.ws.send(json.dumps({"t": "sg"}))
        while(self.state == None):
            await asyncio.sleep(1)

    async def receive_data(self):
       while self.ws is not None:
            try:
                result = await self.ws.recv()
                if result and result != "{\"type\": \"ping\"}":
                    data = json.loads(result)
                    self.state = self.deserialize_state(data)
                    #logger.info(f"Received state: {reward1}, {reward2}")
            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket connection is closed: {e}")
                await self.create_connection()
            #except Exception as e:
            #    logger.error(f"Error receiving data: {result}")
            
    async def restart_game(self):
        self.state = None
        await self.ws.send(json.dumps({"t": "sg"}))
        while(self.state == None):
            await asyncio.sleep(1)
        #await self.ws.send(json.dumps({"t": "restart_game"}))
        
    async def send_action(self, action_p1, action_p2):
        if self.ws is not None:
            if action_p1 == 1:
                action_p1 = 0
            elif action_p1 == 2:
                action_p1 = 7
            else:
                action_p1 = -7
            if action_p2 == 1:
                action_p2 = 0
            elif action_p2 == 2:
                action_p2 = 7
            else:
                action_p2 = -7
            try:
                await self.ws.send(json.dumps({"t": "pi", "p1": action_p1, "p2": action_p2}))
                await self.ws.send(json.dumps({"t": "pi", "p1": 0, "p2": 0}))
                #logger.info(f"Sent action p1, p2: {action_p1}, {action_p2}")
            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket connection is closed: {e}")
                await self.reconnect_and_retry(action_p1, action_p2)
        else:
            logger.warning("WebSocket is not connected")
            await self.create_connection()
            if self.ws is not None:
                await self.send_action(action_p1, action_p2)
    
    async def reconnect_and_retry(self, action_p1, action_p2):
        await self.create_connection()
        if self.ws is not None:
            await self.send_action(action_p1, action_p2)

    def deserialize_state(self, data):
        #print(f"Data: {data}")
        ball_x = data['ball']['x']
        ball_y = data['ball']['y']
        ball_vx = data['ball']['vx']
        ball_vy = data['ball']['vy']
        paddle1_y = data['p1']['y']
        paddle2_y = data['p2']['y']
        #print(f"Padel1: {paddle1_y}, paddle2: {paddle2_y}")
        score1 = data['s1']
        score2 = data['s2']
        game_over = data['go']
        r_1 = 0
        r_2 = 0
        #print(f"vx: {ball_vx}, vy: {ball_vy}")
        #print(f" SElf score 1: {self.score1}, score 2: {self.score2}")
        #print(f" SElf r_1: {self.reward1}, r_2: {self.reward2}")
        if score2 > self.score2:
            print(f"PLayer 2 got a point")
            #self.score2 = score2
            r_1 = -2
            r_2 = 2
        if score1 > self.score1:
            #print(f"update score1")
            print(f"PLayer 1 got a point")
            #self.score1 = score1
            r_1 = 2
            r_2 = -2
        if score1 == 50:
            print(f"PLayer 1 won")
            r_1 = 10
            r_2 = -10
        if score2 == 50:
            print(f"PLayer 2 won")
            r_1 = -10
            r_2 = 10
        if ball_x >= 620.0 and ball_y <= paddle2_y + 70.0 and ball_y >= paddle2_y:
            r_2 = 1
        if ball_x <= 20.0  and ball_y <= paddle1_y + 70.0 and ball_y >= paddle1_y:
            r_1 = 1
        if self.state != None and paddle1_y == self.state[0] * 290.0:
            self.blocked1 += 1
            if self.blocked1 > 1000:
                print("player 1 Blocked")
                r_1 -= 1
                self.blocked1 = 0
        if self.state != None and paddle2_y == self.state[1] * 290.0:
            self.blocked2 += 1
            if self.blocked2 > 1000:
                print("player 2 Blocked")
                r_2 -= -1
                self.blocked2 = 0
        done = game_over
        self.reward1 = r_1
        self.reward2 = r_2
        self.score1 = score1
        self.score2 = score2
        if r_1 == 2 or r_1 == -2:
            return self.state
        self.done = done
        return (paddle1_y / 290.0, paddle2_y / 290.0, ball_x /640.0, ball_y /360.0, ball_vx / 15.0, ball_vy / 15.0)
    

    async def close_connection(self):
        if self.ws is not None:
            await self.ws.close()
            print("WebSocket connection closed")
